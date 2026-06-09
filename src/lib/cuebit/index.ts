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

/**
 * 한 프레임 추론에 필요한 버퍼 세트
 */
interface BufferSet {
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
		// 디버깅용: 검출된 선분의 끝점들
		const lines: Line<"fetch">[] = [];
		if (maxIdx >= 0) {
			// 무한직선을 위의 두 점(start, end)으로 표현. 방향은 end-start로 유도.
			const lineDir = (l: Line<"fetch">) => ({
				px: l.start.x,
				py: l.start.y,
				vx: l.end.x - l.start.x,
				vy: l.end.y - l.start.y,
			});

			// 1) 가장 큰 컨투어의 외곽선만 그린 엣지 이미지 생성 (노이즈 블롭 제외).
			const edge = track(cv.Mat.zeros(roi.rows, roi.cols, cv.CV_8UC1));
			cv.drawContours(edge, contours, maxIdx, new cv.Scalar(255), 1);

			// 2) HoughLinesP로 당구대 변의 "가시 구간"을 직선 선분으로 검출.
			//    꼭짓점이 가려져도 변 자체는 직선 선분으로 검출되므로,
			//    뒤에서 직선을 교차시켜 가려진 꼭짓점을 복원할 수 있음.
			const tableScale = Math.min(roi.rows, roi.cols);
			const linesMat = track(new cv.Mat());
			cv.HoughLinesP(
				edge,
				linesMat,
				// rho(px), theta(rad) 해상도
				1,
				Math.PI / 180,
				// threshold: 직선으로 인정할 최소 투표 수 (튜닝값)
				Math.floor(tableScale * 0.01),
				// minLineLength: 너무 짧은 선분은 노이즈로 간주 (튜닝값)
				tableScale * 0.15,
				// maxLineGap: 변 위의 끊긴 구간을 이어붙이는 허용 간격 (튜닝값)
				tableScale * 0.1,
			);

			// region 중심을 ROI 좌표계로 변환 (변을 상/하/좌/우로 나누는 기준점)
			const ccx = cx - x0;
			const ccy = cy - y0;

			type Seg = {
				x1: number;
				y1: number;
				x2: number;
				y2: number;
				mx: number;
				my: number;
				ang: number;
				len: number;
			};
			const segs: Seg[] = [];
			for (let i = 0; i < linesMat.rows; i++) {
				const x1 = linesMat.data32S[i * 4];
				const y1 = linesMat.data32S[i * 4 + 1];
				const x2 = linesMat.data32S[i * 4 + 2];
				const y2 = linesMat.data32S[i * 4 + 3];
				const dx = x2 - x1;
				const dy = y2 - y1;
				const len = Math.hypot(dx, dy);

				lines.push({
					start: { x: x1 + x0, y: y1 + y0 },
					end: { x: x2 + x0, y: y2 + y0 },
				});
				if (len < 1e-3) {
					continue;
				}
				// 방향각을 [0, π)로 정규화 (선분은 방향이 없으므로 180° 주기)
				let ang = Math.atan2(dy, dx);
				if (ang < 0) {
					ang += Math.PI;
				}
				segs.push({
					x1,
					y1,
					x2,
					y2,
					mx: (x1 + x2) / 2,
					my: (y1 + y2) / 2,
					ang,
					len,
				});
			}

			// 중심 기준 부호 오프셋(off)을 함께 들고 다니는 선분
			// off의 절댓값 = "중심에서 얼마나 바깥쪽인지" → 안쪽 선분을 거를 때 사용
			type GSeg = Seg & { off: number };
			// groups[0,1]: ref와 평행한 두 변 / groups[2,3]: 수직인 두 변
			const groups: [GSeg[], GSeg[], GSeg[], GSeg[]] = [[], [], [], []];

			if (segs.length > 0) {
				// 가장 긴 선분의 방향을 기준 방향(ref)으로 사용.
				let ref = 0;
				let maxLen = -1;
				for (const s of segs) {
					if (s.len > maxLen) {
						maxLen = s.len;
						ref = s.ang;
					}
				}
				const ux = Math.cos(ref);
				const uy = Math.sin(ref);

				// [0, π) 원형 거리
				const circDist = (a: number, b: number) => {
					const d = Math.abs(a - b) % Math.PI;
					return Math.min(d, Math.PI - d);
				};

				for (const s of segs) {
					let idx: number;
					let off: number;
					if (circDist(s.ang, ref) < Math.PI / 4) {
						// ref와 평행한 변(예: 상/하) → ref 법선 방향 부호로 둘로 분리
						off = (s.mx - ccx) * -uy + (s.my - ccy) * ux;
						idx = off >= 0 ? 0 : 1;
					} else {
						// ref와 수직인 변(예: 좌/우) → ref 방향 부호로 둘로 분리
						off = (s.mx - ccx) * ux + (s.my - ccy) * uy;
						idx = off >= 0 ? 2 : 3;
					}
					groups[idx].push({ ...s, off });
				}
			}

			// 가중치 붙은 점 (선분 길이를 가중치로 사용)
			type WPoint = { x: number; y: number; w: number };

			// 3) 각 변에 속한 점들로 가중 직선 피팅 (total least squares).
			const fitLine = (pts: WPoint[]): Line<"fetch"> | null => {
				let W = 0;
				let mx = 0;
				let my = 0;
				for (const p of pts) {
					W += p.w;
					mx += p.w * p.x;
					my += p.w * p.y;
				}
				if (W <= 0 || pts.length < 2) {
					return null;
				}
				mx /= W;
				my /= W;

				let sxx = 0;
				let sxy = 0;
				let syy = 0;
				for (const p of pts) {
					const dx = p.x - mx;
					const dy = p.y - my;
					sxx += p.w * dx * dx;
					sxy += p.w * dx * dy;
					syy += p.w * dy * dy;
				}
				// 2x2 공분산행렬의 주축(최대 고유벡터) 방향
				const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
				// 무게중심(start)과 주축 방향 단위벡터만큼 떨어진 점(end)으로 직선 표현
				return {
					start: { x: mx, y: my },
					end: { x: mx + Math.cos(theta), y: my + Math.sin(theta) },
				};
			};

			// 각 변 그룹에서 "가장 바깥(중심에서 가장 먼)" 선분 밴드만 남겨 피팅.
			//    안쪽 쿠션/반사로 검출된 선분은 |off|가 작아 자동으로 제외됨.
			const OUTER_BAND = tableScale * 0.05; // 바깥 밴드 두께 (튜닝값)
			const fitOuter = (gsegs: GSeg[]): Line<"fetch"> | null => {
				if (gsegs.length === 0) {
					return null;
				}
				let maxAbs = 0;
				for (const g of gsegs) {
					maxAbs = Math.max(maxAbs, Math.abs(g.off));
				}
				const pts: WPoint[] = [];
				for (const g of gsegs) {
					if (Math.abs(g.off) < maxAbs - OUTER_BAND) {
						continue;
					}
					// 끝점 두 개를 길이 가중치와 함께 추가
					pts.push({ x: g.x1, y: g.y1, w: g.len });
					pts.push({ x: g.x2, y: g.y2, w: g.len });
				}
				return fitLine(pts);
			};

			const candidateLines = groups.map(fitOuter);

			// 4) 4변이 모두 검출된 경우에만 사각형 복원.
			//    (변 하나가 통째로 가려지면 복원 불가 → 다음 프레임에서 재시도)
			if (candidateLines.every((l): l is Line<"fetch"> => l !== null)) {
				logger.debug("All 4 lines detected, fitting quad...");

				const fitted = candidateLines as [
					Line<"fetch">,
					Line<"fetch">,
					Line<"fetch">,
					Line<"fetch">,
				];

				// 중심에서 각 직선에 내린 수선의 발 방향으로 4변을 원형 정렬.
				const ordered = fitted
					.map((l) => {
						const { px, py, vx, vy } = lineDir(l);
						const t = (ccx - px) * vx + (ccy - py) * vy;
						const fx = px + t * vx - ccx;
						const fy = py + t * vy - ccy;
						return { line: l, angle: Math.atan2(fy, fx) };
					})
					.sort((a, b) => a.angle - b.angle)
					.map((o) => o.line);

				// 5) 인접한 두 직선의 교점을 꼭짓점으로 사용.
				const intersect = (
					a: Line<"fetch">,
					b: Line<"fetch">,
				): { x: number; y: number } => {
					const la = lineDir(a);
					const lb = lineDir(b);
					const denom = la.vx * lb.vy - la.vy * lb.vx;
					// 거의 평행하면 교점이 불안정 → 두 기준점의 중점으로 대체
					if (Math.abs(denom) < 1e-6) {
						return { x: (la.px + lb.px) / 2, y: (la.py + lb.py) / 2 };
					}
					const t = ((lb.px - la.px) * lb.vy - (lb.py - la.py) * lb.vx) / denom;
					return { x: la.px + t * la.vx, y: la.py + t * la.vy };
				};

				const corners = ordered.map((_, i) => {
					const c = intersect(ordered[i], ordered[(i + 1) % 4]);
					return { x: c.x + x0, y: c.y + y0 };
				});

				points = [corners[0], corners[1], corners[2], corners[3]];

				logger.debug(`Fitted quad points:`);
				for (const [i, point] of points.entries()) {
					logger.debug(
						`  Point ${i}: (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`,
					);
				}
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
	public async process(source: HTMLVideoElement) {
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
