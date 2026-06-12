import cv from "@techstark/opencv-js";
import type { InferenceSession } from "onnxruntime-web";
import * as ort from "onnxruntime-web/webgpu";
import { alignTo16, dist, measure, snapshotMat, withMatScope } from "@/common";
import logger from "@/lib/logger";
import type { ONNX } from "@/lib/onnx";
import type { FrameInfo } from "../capture";
import hwc2chwShader from "./shaders/hwc2chw.wgsl";
import maskShader from "./shaders/mask.wgsl";
import resizeShader from "./shaders/resize.wgsl";

/**
 * 버퍼 인덱스
 */
export type BufferIndex = 0 | 1;

export type FrameResult = {
	readonly table:
		| {
				transform: {
					quad: Quad<"fetch">;
					matrix: {
						transform: MatSnapshot;
						inverseTransform: MatSnapshot;
					};
				};
				approximation: TableApproximation;
		  }
		| {
				approximation: TableApproximation;
				transform?: undefined;
		  }
		| null;
	readonly ballPoints: Vector2<"fetch">[];
	readonly cue: {
		approximation: CueApproximation;
	} | null;
};

/**
 * 한 프레임 추론에 필요한 버퍼 세트
 */
export interface BufferSet {
	readonly resizePipeline: GPUComputePipeline;
	readonly preprocessPipeline: GPUComputePipeline;
	/**
	 * 프레임을 복사할 텍스처
	 */
	readonly frameTexture: GPUTexture;
	/**
	 * 리사이즈된 프레임 텍스쳐
	 */
	readonly resizedFrameTexture: GPUTexture;
	/**
	 * 리사이즈된 프레임을 저장할 텍스처
	 */
	readonly resizeBindGroup: GPUBindGroup;
	/**
	 * 셰이더에서 프레임 데이터를 읽어올 때 사용하는 바인드 그룹
	 */
	readonly preprocessBindGroup: GPUBindGroup;
	/**
	 * 셰이더에서 프레임 데이터를 읽어올 버퍼
	 */
	readonly inputBuffer: GPUBuffer;
	/**
	 * ONNX Runtime에서 GPU 버퍼를 텐서로 사용할 때 필요한 래퍼 객체
	 */
	readonly inputTensor: ort.Tensor;
	/**
	 * 모델의 첫 번째 출력 버퍼
	 */
	readonly detectionsBuffer: GPUBuffer;
	/**
	 * 모델의 첫 번째 출력 텐서
	 */
	readonly detectionsTensor: ort.Tensor;
	/**
	 * 모델의 두 번째 출력 버퍼
	 */
	readonly protosBuffer: GPUBuffer;
	/**
	 * 모델의 두 번째 출력 텐서
	 */
	readonly protosTensor: ort.Tensor;
	/**
	 * output0를 CPU에 전달하기 위한 staging 버퍼
	 */
	readonly detectionsReadBuffer: GPUBuffer;
	/**
	 * 현재 버퍼에 대해 진행 중인 추론 결과를 나타내는 Promise
	 */
	pendingSegmentationInference: Promise<InferenceSession.OnnxValueMapType> | null;

	readonly maskPipeline: GPUComputePipeline;
	/**
	 * 마스크 생성 셰이더에서 사용할 바인드 그룹
	 */
	readonly maskBindgroup: GPUBindGroup;
	/**
	 * 마스크 생성 셰이더에서 사용할 Params 버퍼
	 */
	readonly maskCandidateIndexBuffer: GPUBuffer;
	/**
	 * 마스크 이미지를 저장하는 버퍼
	 */
	readonly maskBuffer: GPUBuffer;
	/**
	 * 마스크 이미지의 프레임 텍스처
	 */
	readonly maskFrameTexture: GPUTexture;
	/**
	 * 테이블 마스크 이미지를 저장하는 버퍼
	 * 디버깅용
	 */
	readonly tableMaskFrameTexture: GPUTexture;
	/**
	 * 큐 마스크 이미지를 저장하는 버퍼
	 * 디버깅용
	 */
	readonly cueMaskFrameTexture: GPUTexture;
	/**
	 * 마스크 이미지를 CPU에 전달하기 위한 staging 버퍼:
	 */
	readonly tableMaskReadBuffer: GPUBuffer;
	/**
	 * 큐 마스크 이미지를 CPU에 전달하기 위한 staging 버퍼
	 */
	readonly cueMaskReadBuffer: GPUBuffer;
}

type DetectionMask = {
	readonly detection: Detection;
	readonly mask: Float32Array;
};

type Postprocess = {
	readonly tableMask: DetectionMask | null;
	readonly balls: Vector2<"feed">[];
	readonly cueMask: DetectionMask | null;
};

function toQuad<S extends VectorSpace>(
	points: [Vector2<S>, Vector2<S>, Vector2<S>, Vector2<S>],
): Quad<S> {
	const indexedPoints = points.map((point, index) => ({
		point,
		index,
	}));

	const leftmost = indexedPoints.reduce((acc, cur) =>
		acc.point.x < cur.point.x ? acc : cur,
	);
	const rightmost = indexedPoints[(leftmost.index + 2) % 4];

	const topmost = indexedPoints.reduce((acc, cur) =>
		acc.point.y < cur.point.y ? acc : cur,
	);
	const bottommost = indexedPoints[(topmost.index + 2) % 4];

	if (
		dist(leftmost.point, topmost.point) > dist(leftmost.point, bottommost.point)
	) {
		return {
			points: {
				topLeft: bottommost.point,
				bottomLeft: leftmost.point,
				bottomRight: topmost.point,
				topRight: rightmost.point,
			},
		};
	}

	return {
		points: {
			topLeft: leftmost.point,
			bottomLeft: topmost.point,
			bottomRight: rightmost.point,
			topRight: bottommost.point,
		},
	};
}

function getTransformMatrix<S extends VectorSpace>(quad: Quad<S>) {
	return withMatScope((track) => {
		const src = track(
			cv.matFromArray(4, 1, cv.CV_32FC2, [
				quad.points.topLeft.x ?? 0,
				quad.points.topLeft.y ?? 0,
				quad.points.bottomLeft.x ?? 0,
				quad.points.bottomLeft.y ?? 0,
				quad.points.bottomRight.x ?? 0,
				quad.points.bottomRight.y ?? 0,
				quad.points.topRight.x ?? 0,
				quad.points.topRight.y ?? 0,
			]),
		);
		const dst = track(
			cv.matFromArray(
				4,
				1,
				cv.CV_32FC2,
				[
					// Top-Left
					0, 1422,
					// Bottom-Left
					0, 0,
					// Bottom-Right
					2844, 0,
					// Top-Right
					2844, 1422,
				],
			),
		);
		const transform = track(cv.getPerspectiveTransform(src, dst));
		const inverseTransform = track(transform.inv(cv.DECOMP_LU));

		return {
			transform: snapshotMat(transform),
			inverseTransform: snapshotMat(inverseTransform),
		};
	});
}

type TableApproximation = {
	readonly mask: DetectionMask;
	readonly points:
		| [Vector2<"fetch">, Vector2<"fetch">, Vector2<"fetch">, Vector2<"fetch">]
		| null;
	readonly lines: Line<"fetch">[];
};

function findTableQuad(
	tableMask: DetectionMask,
	width: number,
	height: number,
	region: BoundingBox<"fetch">,
): TableApproximation {
	return withMatScope((track) => {
		// Float32 → 0/255 binary Mat
		const src = track(new cv.Mat(height, width, cv.CV_8UC1));
		for (let i = 0; i < width * height; i++) {
			src.data[i] = tableMask.mask[i] > 0.5 ? 255 : 0;
		}
		const cx = (region.lt.x + region.rb.x) / 2;
		const cy = (region.lt.y + region.rb.y) / 2;
		const hw = ((region.rb.x - region.lt.x) / 2) * 1.2; // MARGIN ≈ 1.1~1.2
		const hh = ((region.rb.y - region.lt.y) / 2) * 1.2;

		const x0 = Math.max(0, Math.floor(cx - hw));
		const y0 = Math.max(0, Math.floor(cy - hh));
		const x1 = Math.min(width, Math.ceil(cx + hw));
		const y1 = Math.min(height, Math.ceil(cy + hh));
		const roi = track(src.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0)));

		// 마스크의 작은 구멍(빵꾸)을 메움: morphological closing (팽창 후 침식).
		// 구멍이 컨투어를 끊어 잘못된 변이 잡히는 것을 방지.
		const closeSize =
			(Math.max(3, Math.round(Math.min(roi.rows, roi.cols) * 0.03)) | 1) >>> 0;
		const kernel = track(
			cv.getStructuringElement(
				cv.MORPH_ELLIPSE,
				new cv.Size(closeSize, closeSize),
			),
		);
		cv.morphologyEx(roi, roi, cv.MORPH_CLOSE, kernel);

		const contours = track(new cv.MatVector());
		cv.findContours(
			roi,
			contours,
			// 현재는 계층 정보가 필요하지 않지만, 인터페이스가 필요로 함
			track(new cv.Mat()),
			// 가장 바깥 외곽선만 검출
			cv.RETR_EXTERNAL,
			// 직선 구간은 양 끝점만 저장
			cv.CHAIN_APPROX_SIMPLE,
		);

		// 면적 가장 큰 컨투어
		let maxArea = 0;
		let maxIdx = -1;
		for (let i = 0; i < contours.size(); i++) {
			const area = cv.contourArea(contours.get(i));
			if (area > maxArea) {
				maxArea = area;
				maxIdx = i;
			}
		}

		let points:
			| [Vector2<"fetch">, Vector2<"fetch">, Vector2<"fetch">, Vector2<"fetch">]
			| null = null;
		// 디버깅용: 피팅된 4개의 변(직선)
		const lines: Line<"fetch">[] = [];
		if (maxIdx >= 0) {
			const contour = contours.get(maxIdx);
			const tableScale = Math.min(roi.rows, roi.cols);

			// 점, 그리고 점 + 단위방향으로 표현한 무한직선 (ROI 좌표계, 내부 계산용)
			type P = { x: number; y: number };
			type ILine = { px: number; py: number; vx: number; vy: number };

			// contour 점들을 배열로 추출
			const cpts: P[] = [];
			for (let i = 0; i < contour.rows; i++) {
				cpts.push({
					x: contour.data32S[i * 2],
					y: contour.data32S[i * 2 + 1],
				});
			}

			// total least squares 직선 피팅
			const tlsFit = (pts: P[]): ILine | null => {
				if (pts.length < 2) {
					return null;
				}
				let mx = 0;
				let my = 0;
				for (const p of pts) {
					mx += p.x;
					my += p.y;
				}
				mx /= pts.length;
				my /= pts.length;
				let sxx = 0;
				let sxy = 0;
				let syy = 0;
				for (const p of pts) {
					const dx = p.x - mx;
					const dy = p.y - my;
					sxx += dx * dx;
					sxy += dx * dy;
					syy += dy * dy;
				}
				// 2x2 공분산행렬의 주축(최대 고유벡터) 방향
				const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
				return { px: mx, py: my, vx: Math.cos(theta), vy: Math.sin(theta) };
			};

			// RANSAC 직선 피팅: 내부선/가림 부분/노이즈를 outlier로 제거.
			const RANSAC_THRESHOLD = Math.max(2, tableScale * 0.02); // 인라이어 허용 거리(px)
			const RANSAC_ITERS = 80;
			const ransacFit = (pts: P[]): ILine | null => {
				if (pts.length < 2) {
					return null;
				}
				if (pts.length === 2) {
					return tlsFit(pts);
				}
				let bestAx = 0;
				let bestAy = 0;
				let bestNx = 0;
				let bestNy = 0;
				let bestCount = -1;
				for (let k = 0; k < RANSAC_ITERS; k++) {
					const i = Math.floor(Math.random() * pts.length);
					let j = Math.floor(Math.random() * pts.length);
					if (i === j) {
						j = (j + 1) % pts.length;
					}
					const a = pts[i];
					const b = pts[j];
					let dx = b.x - a.x;
					let dy = b.y - a.y;
					const len = Math.hypot(dx, dy);
					if (len < 1e-6) {
						continue;
					}
					dx /= len;
					dy /= len;
					// 후보 직선의 법선
					const nx = -dy;
					const ny = dx;
					let count = 0;
					for (const p of pts) {
						if (
							Math.abs((p.x - a.x) * nx + (p.y - a.y) * ny) < RANSAC_THRESHOLD
						) {
							count++;
						}
					}
					if (count > bestCount) {
						bestCount = count;
						bestAx = a.x;
						bestAy = a.y;
						bestNx = nx;
						bestNy = ny;
					}
				}
				if (bestCount < 2) {
					return null;
				}
				// 최적 모델의 인라이어만 모아 정밀 재피팅(TLS)
				const inliers = pts.filter(
					(p) =>
						Math.abs((p.x - bestAx) * bestNx + (p.y - bestAy) * bestNy) <
						RANSAC_THRESHOLD,
				);
				return tlsFit(inliers);
			};

			// 1) 대략적 4변(원근 사각형): convex hull → approxPolyDP epsilon 이분탐색으로
			//    정확히 4점까지 단순화. 점 배정의 "기준"으로만 사용 (정밀도는 RANSAC이 담당).
			const hull = track(new cv.Mat());
			cv.convexHull(contour, hull, false, true);
			const roughCorners = ((): P[] => {
				const peri = cv.arcLength(hull, true);
				let lo = 0;
				let hi = peri;
				for (let it = 0; it < 32; it++) {
					const mid = (lo + hi) / 2;
					const approx = track(new cv.Mat());
					cv.approxPolyDP(hull, approx, mid, true);
					const n = approx.rows;
					if (n === 4) {
						return [0, 1, 2, 3].map((i) => ({
							x: approx.data32S[i * 2],
							y: approx.data32S[i * 2 + 1],
						}));
					}
					// epsilon이 커질수록 꼭짓점 수는 단조 감소
					if (n > 4) {
						lo = mid;
					} else {
						hi = mid;
					}
				}
				// 4점을 못 얻으면 x±y 극값으로 추정 (fallback)
				let tl = cpts[0];
				let br = cpts[0];
				let tr = cpts[0];
				let bl = cpts[0];
				for (const p of cpts) {
					if (p.x + p.y < tl.x + tl.y) tl = p;
					if (p.x + p.y > br.x + br.y) br = p;
					if (p.x - p.y > tr.x - tr.y) tr = p;
					if (p.x - p.y < bl.x - bl.y) bl = p;
				}
				return [tl, tr, br, bl];
			})();

			// 2) contour 점들을 가장 가까운(수직거리) 대략적 변에 배정.
			const groups: [P[], P[], P[], P[]] = [[], [], [], []];
			for (const p of cpts) {
				let best = 0;
				let bestDist = Infinity;
				for (let e = 0; e < 4; e++) {
					const a = roughCorners[e];
					const b = roughCorners[(e + 1) % 4];
					let vx = b.x - a.x;
					let vy = b.y - a.y;
					const len = Math.hypot(vx, vy) || 1;
					vx /= len;
					vy /= len;
					// 변의 무한직선까지의 수직거리
					const d = Math.abs((p.x - a.x) * -vy + (p.y - a.y) * vx);
					if (d < bestDist) {
						bestDist = d;
						best = e;
					}
				}
				groups[best].push(p);
			}

			// 3) 각 변을 RANSAC로 robust 피팅.
			const fittedLines = groups.map(ransacFit);

			// 디버깅용: 피팅된 직선을 긴 선분으로 변환해 보관 (fetch 좌표)
			for (const l of fittedLines) {
				if (!l) {
					continue;
				}
				lines.push({
					start: {
						x: l.px - l.vx * tableScale + x0,
						y: l.py - l.vy * tableScale + y0,
					},
					end: {
						x: l.px + l.vx * tableScale + x0,
						y: l.py + l.vy * tableScale + y0,
					},
				});
			}

			// 4) 4변이 모두 피팅된 경우에만 인접 직선 교차로 꼭짓점 복원.
			//    (변 하나가 통째로 가려지면 복원 불가 → 다음 프레임에서 재시도)
			if (fittedLines.every((l): l is ILine => l !== null)) {
				const fitted = fittedLines as [ILine, ILine, ILine, ILine];

				const intersect = (a: ILine, b: ILine): P => {
					const denom = a.vx * b.vy - a.vy * b.vx;
					// 거의 평행하면 교점이 불안정 → 두 기준점의 중점으로 대체
					if (Math.abs(denom) < 1e-6) {
						return { x: (a.px + b.px) / 2, y: (a.py + b.py) / 2 };
					}
					const t = ((b.px - a.px) * b.vy - (b.py - a.py) * b.vx) / denom;
					return { x: a.px + t * a.vx, y: a.py + t * a.vy };
				};

				// roughCorners가 순환 순서이므로 변(fitted)도 순환 순서.
				// 변 i와 변 (i+1)의 교점이 곧 인접 꼭짓점.
				const corners = fitted.map((_, i) => {
					const c = intersect(fitted[i], fitted[(i + 1) % 4]);
					return { x: c.x + x0, y: c.y + y0 };
				});

				points = [corners[0], corners[1], corners[2], corners[3]];
			}
		}

		return {
			mask: tableMask,
			points,
			lines,
		};
	});
}

type CueApproximation = {
	mask: DetectionMask;
	endpoints: [Vector2<"fetch">, Vector2<"fetch">] | null;
};

function findCue(
	cueMask: DetectionMask,
	width: number,
	height: number,
	region: BoundingBox<"fetch">,
): CueApproximation {
	return withMatScope((track) => {
		// Float32 → 0/255 binary Mat
		const src = track(new cv.Mat(height, width, cv.CV_8UC1));
		for (let i = 0; i < width * height; i++) {
			src.data[i] = cueMask.mask[i] > 0.5 ? 255 : 0;
		}

		const x0 = Math.max(0, Math.floor(region.lt.x));
		const y0 = Math.max(0, Math.floor(region.lt.y));
		const x1 = Math.min(width, Math.ceil(region.rb.x));
		const y1 = Math.min(height, Math.ceil(region.rb.y));

		const roi = track(src.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0)));

		// 가장 큰 컨투어만 사용 (노이즈 덩어리 제거).
		// NOTE: 큐는 매우 가는 선이라 morphology open을 적용하면 통째로 지워지므로 쓰지 않음.
		const contours = track(new cv.MatVector());
		cv.findContours(
			roi,
			contours,
			track(new cv.Mat()),
			cv.RETR_EXTERNAL,
			cv.CHAIN_APPROX_SIMPLE,
		);

		const result: CueApproximation = {
			mask: cueMask,
			endpoints: null,
		};

		let maxArea = 0;
		let maxIdx = -1;
		for (let i = 0; i < contours.size(); i++) {
			const area = cv.contourArea(contours.get(i));
			if (area > maxArea) {
				maxArea = area;
				maxIdx = i;
			}
		}
		if (maxIdx >= 0) {
			const contour = contours.get(maxIdx);

			// 큐의 주축(principal axis)을 직접 피팅.
			// HoughLinesP는 채워진 마스크의 "테두리"를 검출해 중심축에서 벗어나지만,
			// fitLine은 덩어리 전체에 직선을 피팅하므로 중심축을 곧바로 얻을 수 있음.
			// DIST_HUBER로 이상치(노이즈 픽셀)에 강건하게 피팅.
			const lineParams = track(new cv.Mat());
			cv.fitLine(contour, lineParams, cv.DIST_HUBER, 0, 0.01, 0.01);
			const vx = lineParams.data32F[0];
			const vy = lineParams.data32F[1];
			const px = lineParams.data32F[2];
			const py = lineParams.data32F[3];

			// 컨투어 점들을 축 방향으로 투영해 양 끝점(투영 최소/최대)을 찾음
			let minT = Infinity;
			let maxT = -Infinity;
			for (let i = 0; i < contour.rows; i++) {
				const cxi = contour.data32S[i * 2];
				const cyi = contour.data32S[i * 2 + 1];
				const t = (cxi - px) * vx + (cyi - py) * vy;
				if (t < minT) {
					minT = t;
				}
				if (t > maxT) {
					maxT = t;
				}
			}
			if (Number.isFinite(minT) && Number.isFinite(maxT)) {
				result.endpoints = [
					{
						x: px + minT * vx + x0,
						y: py + minT * vy + y0,
					},
					{
						x: px + maxT * vx + x0,
						y: py + maxT * vy + y0,
					},
				];
			}
		}

		return result;
	});
}

class Detection {
	public readonly index: number;
	public readonly bbox: BoundingBox<"feed">;
	public readonly confidence: number;
	public readonly classId: number;
	public readonly coefficients: Float32Array;

	constructor(index: number, chunk: Float32Array) {
		this.index = index;
		this.bbox = {
			lt: {
				x: chunk[0],
				y: chunk[1],
			},
			rb: {
				x: chunk[2],
				y: chunk[3],
			},
		};
		this.confidence = chunk[4];
		this.classId = chunk[5];
		this.coefficients = chunk.subarray(6);
	}
}

function toDetections(detection: Float32Array, chunkSize: number): Detection[] {
	const detections: Detection[] = [];

	for (let i = 0; i < detection.length; i++) {
		const offset = i * chunkSize;
		detections.push(
			new Detection(i, detection.subarray(offset, offset + chunkSize)),
		);
	}

	return detections;
}

/**
 * 전체 파이프라인 실행 클래스
 */
class Cuebit {
	private readonly device: GPUDevice;
	private readonly onnx: ONNX;
	private readonly frameInfo: FrameInfo;
	private readonly preprocessShaderModule: GPUShaderModule;
	private readonly maskShaderModule: GPUShaderModule;
	private readonly buffers: [BufferSet, BufferSet];
	private currentBufferIndex: BufferIndex = 0;

	constructor(device: GPUDevice, onnx: ONNX, frameInfo: FrameInfo) {
		this.device = device;
		this.onnx = onnx;
		this.frameInfo = frameInfo;
		this.preprocessShaderModule = device.createShaderModule({
			code: hwc2chwShader,
		});
		this.maskShaderModule = device.createShaderModule({
			code: maskShader,
		});

		this.buffers = [
			this.createBufferSet(frameInfo, onnx),
			this.createBufferSet(frameInfo, onnx),
		];
	}

	private createBufferSet(frameInfo: FrameInfo, onnx: ONNX): BufferSet {
		const resizePipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.device.createShaderModule({
					code: resizeShader,
				}),
				entryPoint: "resize",
			},
		});
		const preprocessPipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.preprocessShaderModule,
				entryPoint: "hwc2chw",
			},
		});
		const frameTexture = this.device.createTexture({
			size: [frameInfo.width, frameInfo.height],
			format: "rgba8unorm",
			usage:
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.RENDER_ATTACHMENT,
		});

		const resizedFrameTexture = this.device.createTexture({
			size: [
				onnx.segementation.input.feeds.image.width,
				onnx.segementation.input.feeds.image.height,
			],
			format: "rgba8unorm",
			usage:
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.STORAGE_BINDING,
		});

		const sampler = this.device.createSampler({
			magFilter: "linear",
			minFilter: "linear",
		});

		const resizeParamsBuffer = this.device.createBuffer({
			label: "Resize Params Buffer",
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			size: alignTo16(4 * 4), // width, height, srcWidth, srcHeight
		});
		this.device.queue.writeBuffer(
			resizeParamsBuffer,
			0,
			new Uint32Array([
				frameInfo.width,
				frameInfo.height,
				onnx.segementation.input.feeds.image.width,
				onnx.segementation.input.feeds.image.height,
			]),
		);

		const resizeBindGroup = this.device.createBindGroup({
			layout: resizePipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: frameTexture.createView(),
				},
				{
					binding: 1,
					resource: resizedFrameTexture.createView(),
				},
				{
					binding: 2,
					resource: {
						buffer: resizeParamsBuffer,
					},
				},
				{
					binding: 3,
					resource: sampler,
				},
			],
		});

		const inputBuffer = this.device.createBuffer({
			label: "Input Buffer",
			usage:
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.STORAGE,
			// 4 byte * 3 channel * width * height
			size: alignTo16(4 * onnx.segementation.input.feeds.image.size),
		});

		const inputTensor = ort.Tensor.fromGpuBuffer(inputBuffer, {
			dataType: "float32",
			dims: onnx.segementation.input.feeds.image.shape,
		});

		const preprocessBindGroup = this.device.createBindGroup({
			layout: preprocessPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: resizedFrameTexture.createView(),
				},
				{
					binding: 1,
					resource: {
						buffer: inputBuffer,
					},
				},
			],
		});

		const detectionsBuffer = this.device.createBuffer({
			label: "Detections Buffer",
			usage:
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.STORAGE,
			size: alignTo16(4 * onnx.segementation.output.fetchs.detections.size),
		});
		const detectionsTensor = ort.Tensor.fromGpuBuffer(detectionsBuffer, {
			dataType: "float32",
			dims: onnx.segementation.output.fetchs.detections.shape,
		});

		const protosBuffer = this.device.createBuffer({
			label: "Protos Buffer",
			usage:
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.STORAGE,
			size: alignTo16(4 * onnx.segementation.output.fetchs.protos.size),
		});
		const protosTensor = ort.Tensor.fromGpuBuffer(protosBuffer, {
			dataType: "float32",
			dims: onnx.segementation.output.fetchs.protos.shape,
		});
		const detectionsReadBuffer = this.device.createBuffer({
			label: "Detections Read Buffer",
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			size: detectionsBuffer.size,
		});

		// mask
		const maskPipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.maskShaderModule,
				entryPoint: "createMask",
			},
		});
		const maskCandidateIndexBuffer = this.device.createBuffer({
			label: "Mask Candidate Index Buffer",
			size: alignTo16(1),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
		const maskBuffer = this.device.createBuffer({
			label: "Mask Buffer",
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
			// 4 (float32) byte * width * height
			size: alignTo16(
				4 *
					onnx.segementation.output.fetchs.protos.width *
					onnx.segementation.output.fetchs.protos.height,
			),
		});
		const maskFrameTexture = this.device.createTexture({
			size: [
				onnx.segementation.output.fetchs.protos.width,
				onnx.segementation.output.fetchs.protos.height,
			],
			format: "rgba8unorm",
			usage:
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.STORAGE_BINDING,
		});
		const tableMaskFrameTexture = this.device.createTexture({
			size: [
				onnx.segementation.output.fetchs.protos.width,
				onnx.segementation.output.fetchs.protos.height,
			],
			format: "rgba8unorm",
			usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
		});
		const cueMaskFrameTexture = this.device.createTexture({
			size: [
				onnx.segementation.output.fetchs.protos.width,
				onnx.segementation.output.fetchs.protos.height,
			],
			format: "rgba8unorm",
			usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
		});
		const maskBindgroup = this.device.createBindGroup({
			layout: maskPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {
						buffer: detectionsBuffer,
					},
				},
				{
					binding: 1,
					resource: {
						buffer: protosBuffer,
					},
				},
				{
					binding: 2,
					resource: {
						buffer: maskCandidateIndexBuffer,
					},
				},
				{
					binding: 3,
					resource: {
						buffer: maskBuffer,
					},
				},
				{
					binding: 4,
					resource: maskFrameTexture.createView(),
				},
			],
		});
		const tableMaskReadBuffer = this.device.createBuffer({
			label: "Mask Read Buffer",
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			size: maskBuffer.size,
		});
		const cueMaskReadBuffer = this.device.createBuffer({
			label: "Mask Read Buffer",
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			size: maskBuffer.size,
		});

		return {
			resizePipeline,
			preprocessPipeline,
			frameTexture,
			resizedFrameTexture,
			resizeBindGroup,
			preprocessBindGroup,
			inputBuffer,
			inputTensor,
			detectionsBuffer,
			detectionsTensor,
			protosBuffer,
			protosTensor,
			detectionsReadBuffer,
			pendingSegmentationInference: null,
			maskPipeline,
			maskBindgroup,
			maskCandidateIndexBuffer,
			maskBuffer,
			maskFrameTexture,
			tableMaskFrameTexture,
			cueMaskFrameTexture,
			tableMaskReadBuffer,
			cueMaskReadBuffer,
		};
	}

	/**
	 * 프레임 전처리
	 */
	private preprocessFrame(source: HTMLVideoElement, buffer: BufferSet): void {
		// 프레임을 텍스처로 복사
		this.copyFrameToTexture(source, buffer);

		// 프레임 전처리
		const commandEncoder = this.device.createCommandEncoder();
		this.resize(commandEncoder, buffer);
		this.hwc2chw(commandEncoder, buffer);
		this.device.queue.submit([commandEncoder.finish()]);
	}

	private copyFrameToTexture(
		source: HTMLVideoElement,
		buffer: BufferSet,
	): void {
		// NOTE: importExternalTexture 고려
		this.device.queue.copyExternalImageToTexture(
			{
				source,
			},
			{
				texture: buffer.frameTexture,
			},
			[this.frameInfo.width, this.frameInfo.height],
		);
	}

	private resize(encoder: GPUCommandEncoder, buffer: BufferSet): void {
		const pass = encoder.beginComputePass();
		pass.setPipeline(buffer.resizePipeline);
		pass.setBindGroup(0, buffer.resizeBindGroup);
		pass.dispatchWorkgroups(
			alignTo16(this.onnx.segementation.input.feeds.image.width),
			alignTo16(this.onnx.segementation.input.feeds.image.height),
		);
		pass.end();
	}

	private hwc2chw(encoder: GPUCommandEncoder, buffer: BufferSet): void {
		const pass = encoder.beginComputePass();
		pass.setPipeline(buffer.preprocessPipeline);
		pass.setBindGroup(0, buffer.preprocessBindGroup);
		pass.dispatchWorkgroups(
			alignTo16(this.onnx.segementation.input.feeds.image.width),
			alignTo16(this.onnx.segementation.input.feeds.image.height),
		);
		pass.end();
	}

	private select(
		detections: Detection[],
	): [Detection | null, Detection[], Detection | null] {
		let table: Detection | null = null;
		const balls: Detection[] = [];
		const ballClassIds = new Set([0, 2, 5, 6]);
		let cue: Detection | null = null;

		for (const detection of detections) {
			if (detection.classId === 4) {
				// NOTE: 2: table
				if (table === null || detection.confidence > table.confidence) {
					table = detection;
				}
			} else if (ballClassIds.has(detection.classId)) {
				// NOTE: 0,1,3,4: balls
				if (detection.confidence > 0.25) {
					balls.push(detection);
				}
			} else if (detection.classId === 1) {
				if (cue === null || detection.confidence > cue.confidence) {
					cue = detection;
				}
			}
		}

		return [table, balls, cue];
	}

	private async getDetectionMasks(
		buffer: BufferSet,
		tableDetection: Detection | null,
		cueDetection: Detection | null,
	): Promise<[DetectionMask | null, DetectionMask | null]> {
		// table mask pass
		if (tableDetection) {
			this.device.queue.writeBuffer(
				buffer.maskCandidateIndexBuffer,
				0,
				new Uint32Array([tableDetection.index]),
			);
			const commandEncoder = this.device.createCommandEncoder();
			const tableMaskPass = commandEncoder.beginComputePass();
			tableMaskPass.setPipeline(buffer.maskPipeline);
			tableMaskPass.setBindGroup(0, buffer.maskBindgroup);
			tableMaskPass.dispatchWorkgroups(
				Math.ceil(this.onnx.segementation.output.fetchs.protos.width / 16),
				Math.ceil(this.onnx.segementation.output.fetchs.protos.height / 16),
			);
			tableMaskPass.end();
			commandEncoder.copyBufferToBuffer(
				buffer.maskBuffer,
				0,
				buffer.tableMaskReadBuffer,
				0,
				// 4 byte * width * height
				4 *
					this.onnx.segementation.output.fetchs.protos.width *
					this.onnx.segementation.output.fetchs.protos.height,
			);
			commandEncoder.copyTextureToTexture(
				{
					texture: buffer.maskFrameTexture,
				},
				{
					texture: buffer.tableMaskFrameTexture,
				},
				[
					this.onnx.segementation.output.fetchs.protos.width,
					this.onnx.segementation.output.fetchs.protos.height,
				],
			);

			this.device.queue.submit([commandEncoder.finish()]);
		}

		// cue mask pass
		if (cueDetection) {
			this.device.queue.writeBuffer(
				buffer.maskCandidateIndexBuffer,
				0,
				new Uint32Array([cueDetection.index]),
			);
			const commandEncoder = this.device.createCommandEncoder();
			const cueMaskPass = commandEncoder.beginComputePass();
			cueMaskPass.setPipeline(buffer.maskPipeline);
			cueMaskPass.setBindGroup(0, buffer.maskBindgroup);
			cueMaskPass.dispatchWorkgroups(
				Math.ceil(this.onnx.segementation.output.fetchs.protos.width / 16),
				Math.ceil(this.onnx.segementation.output.fetchs.protos.height / 16),
			);
			cueMaskPass.end();
			commandEncoder.copyBufferToBuffer(
				buffer.maskBuffer,
				0,
				buffer.cueMaskReadBuffer,
				0,
				// 4 byte * width * height
				4 *
					this.onnx.segementation.output.fetchs.protos.width *
					this.onnx.segementation.output.fetchs.protos.height,
			);
			commandEncoder.copyTextureToTexture(
				{
					texture: buffer.maskFrameTexture,
				},
				{
					texture: buffer.cueMaskFrameTexture,
				},
				[
					this.onnx.segementation.output.fetchs.protos.width,
					this.onnx.segementation.output.fetchs.protos.height,
				],
			);
			this.device.queue.submit([commandEncoder.finish()]);
		}

		await Promise.all([
			buffer.tableMaskReadBuffer.mapAsync(GPUMapMode.READ),
			buffer.cueMaskReadBuffer.mapAsync(GPUMapMode.READ),
		]);

		const table: DetectionMask | null = tableDetection && {
			detection: tableDetection,
			mask: new Float32Array(
				buffer.tableMaskReadBuffer.getMappedRange().slice(0),
			),
		};
		buffer.tableMaskReadBuffer.unmap();

		const cue: DetectionMask | null = cueDetection && {
			detection: cueDetection,
			mask: new Float32Array(
				buffer.cueMaskReadBuffer.getMappedRange().slice(0),
			),
		};
		buffer.cueMaskReadBuffer.unmap();

		return [table, cue];
	}

	private async postprocess(buffer: BufferSet): Promise<Postprocess | null> {
		// buffer의 프레임 추론 결과 대기
		await measure(
			() => buffer.pendingSegmentationInference,
			"Pending Inference",
		);

		// buffer의 추론 결과를 staging 버퍼로 복사
		const stagingCommandEncoder = this.device.createCommandEncoder();
		stagingCommandEncoder.copyBufferToBuffer(
			buffer.detectionsBuffer,
			0,
			buffer.detectionsReadBuffer,
			0,
			buffer.detectionsReadBuffer.size,
		);
		this.device.queue.submit([stagingCommandEncoder.finish()]);

		await buffer.detectionsReadBuffer.mapAsync(GPUMapMode.READ);
		const detections = toDetections(
			new Float32Array(buffer.detectionsReadBuffer.getMappedRange().slice(0)),
			this.onnx.segementation.output.fetchs.detections.stride,
		);
		buffer.detectionsReadBuffer.unmap();

		// 추론 결과에서 테이블, 공, 큐 선택
		const [table, balls, cue] = this.select(detections);

		logger.info(
			table
				? `인식된 테이블 index: ${table.index}, confidence: ${table.confidence.toFixed(2)}, bbox: (${table.bbox.lt.x.toFixed(0)}, ${table.bbox.lt.y.toFixed(0)}), (${table.bbox.rb.x.toFixed(0)}, ${table.bbox.rb.y.toFixed(0)})`
				: "테이블 인식 실패",
		);
		logger.info(`인식된 공 ${balls.length}개`);
		logger.info(
			cue
				? `인식된 큐 index: ${cue.index}, confidence: ${cue.confidence.toFixed(2)}, bbox: (${cue.bbox.lt.x.toFixed(0)}, ${cue.bbox.lt.y.toFixed(0)}), (${cue.bbox.rb.x.toFixed(0)}, ${cue.bbox.rb.y.toFixed(0)})`
				: "큐 인식 실패",
		);

		const [tableMask, cueMask] = await this.getDetectionMasks(
			buffer,
			table,
			cue,
		);

		logger.info(
			`테이블 마스크: ${tableMask ? "생성됨" : "생성 실패"}, 큐 마스크: ${cueMask ? "생성됨" : "생성 실패"}`,
		);

		return {
			tableMask,
			balls: balls.map((ball) => ({
				x: (ball.bbox.lt.x + ball.bbox.rb.x) / 2,
				y: (ball.bbox.lt.y + ball.bbox.rb.y) / 2,
			})),
			cueMask,
		};
	}

	private getTablePoints(result: Postprocess) {
		if (!result.tableMask?.mask) {
			return null;
		}

		const feedToFetchScaleX =
			this.onnx.segementation.output.fetchs.protos.width /
			this.onnx.segementation.input.feeds.image.width;
		const feedToFetchScaleY =
			this.onnx.segementation.output.fetchs.protos.height /
			this.onnx.segementation.input.feeds.image.height;

		const approximation = findTableQuad(
			result.tableMask,
			this.onnx.segementation.output.fetchs.protos.width,
			this.onnx.segementation.output.fetchs.protos.height,
			{
				lt: {
					x: result.tableMask.detection.bbox.lt.x * feedToFetchScaleX,
					y: result.tableMask.detection.bbox.lt.y * feedToFetchScaleY,
				},
				rb: {
					x: result.tableMask.detection.bbox.rb.x * feedToFetchScaleX,
					y: result.tableMask.detection.bbox.rb.y * feedToFetchScaleY,
				},
			},
		);

		if (result.tableMask !== null && approximation === null) {
			logger.info("table mask로부터 quad 찾기 실패");
		}

		return approximation;
	}

	private getCuePoints(result: Postprocess) {
		if (!result.cueMask?.mask) {
			return null;
		}

		const feedToFetchScaleX =
			this.onnx.segementation.output.fetchs.protos.width /
			this.onnx.segementation.input.feeds.image.width;
		const feedToFetchScaleY =
			this.onnx.segementation.output.fetchs.protos.height /
			this.onnx.segementation.input.feeds.image.height;

		const approximation = findCue(
			result.cueMask,
			this.onnx.segementation.output.fetchs.protos.width,
			this.onnx.segementation.output.fetchs.protos.height,
			{
				lt: {
					x: result.cueMask.detection.bbox.lt.x * feedToFetchScaleX,
					y: result.cueMask.detection.bbox.lt.y * feedToFetchScaleY,
				},
				rb: {
					x: result.cueMask.detection.bbox.rb.x * feedToFetchScaleX,
					y: result.cueMask.detection.bbox.rb.y * feedToFetchScaleY,
				},
			},
		);

		if (result.cueMask.mask !== null && approximation === null) {
			logger.info("cue mask로부터 큐 끝점 찾기 실패");
		}

		return approximation;
	}

	/**
	 *
	 */
	public async process(source: HTMLVideoElement): Promise<FrameResult> {
		// 이전 버퍼 인덱스 계산
		const previousBufferIndex = 1 - this.currentBufferIndex;

		// 현재 버퍼와 이전 버퍼 참조
		const [currentBuffer, previousBuffer] = [
			this.buffers[this.currentBufferIndex],
			this.buffers[previousBufferIndex],
		];

		this.preprocessFrame(source, currentBuffer);

		const postprocessResult = await measure(
			() => this.postprocess(previousBuffer),
			"Postprocess",
		);

		// 이전 추론이 완료된 후 현재 버퍼에 대해 추론 시작
		currentBuffer.pendingSegmentationInference =
			this.onnx.segementation.session.run(
				{
					[this.onnx.segementation.input.feeds.image.name]:
						currentBuffer.inputTensor,
				},
				{
					[this.onnx.segementation.output.fetchs.detections.name]:
						currentBuffer.detectionsTensor,
					[this.onnx.segementation.output.fetchs.protos.name]:
						currentBuffer.protosTensor,
				},
			);

		const tableApproximation = measure(
			() => postprocessResult && this.getTablePoints(postprocessResult),
			"테이블처럼 보이는 점 찾기",
		);
		const quadForTable =
			(tableApproximation?.points && toQuad(tableApproximation.points)) ?? null;
		const cueApproximation = measure(
			() => postprocessResult && this.getCuePoints(postprocessResult),
			"큐처럼 보이는 점 찾기",
		);

		const transform = quadForTable
			? {
					quad: quadForTable,
					matrix: getTransformMatrix(quadForTable),
				}
			: null;

		// 버퍼 인덱스 업데이트
		this.currentBufferIndex = (1 - this.currentBufferIndex) as BufferIndex;

		const scaleFactorX =
			this.onnx.segementation.output.fetchs.protos.width /
			this.onnx.segementation.input.feeds.image.width;
		const scaleFactorY =
			this.onnx.segementation.output.fetchs.protos.height /
			this.onnx.segementation.input.feeds.image.height;

		const ballPoints: Vector2<"fetch">[] =
			postprocessResult?.balls?.map((ball) => ({
				x: ball.x * scaleFactorX,
				y: ball.y * scaleFactorY,
			})) ?? [];

		if (!postprocessResult) {
			return {
				table: null,
				ballPoints: [],
				cue: null,
			};
		}

		return {
			// TODO: 리팩토링 필요
			table: tableApproximation
				? transform
					? {
							transform,
							approximation: tableApproximation,
						}
					: {
							approximation: tableApproximation,
						}
				: null,
			ballPoints,
			cue: cueApproximation
				? {
						approximation: cueApproximation,
					}
				: null,
		} as const;
	}

	public getCurrentBufferIndex(): BufferIndex {
		return this.currentBufferIndex;
	}

	public getBuffer(bufferIndex: BufferIndex): BufferSet {
		return this.buffers[bufferIndex];
	}
}

export default Cuebit;
