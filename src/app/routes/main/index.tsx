import cv from "@techstark/opencv-js";
import { useEffect, useEffectEvent, useRef, useState } from "react";
import {
	argmin,
	dist,
	measure,
	rerange,
	restoreMat,
	todo,
	withMatScope,
} from "@/common";
import HitControlPanel from "@/components/hit-params-panel";
import OverlayToggleButton from "@/components/overlay-toggle-button";
import hyperparams from "@/config/hyperparams";
import useDebugCanvas from "@/hooks/use-debug-canvas";
import useGPUCanvas from "@/hooks/use-gpu-canvas";
import type { FrameInfo } from "@/lib/capture";
import Cuebit from "@/lib/cuebit";
import logger from "@/lib/logger";
import { device, onnx } from "@/lib/onnx";
import {
	drawTexture,
	TextureTransformer,
	TrajectoryPainter,
} from "@/lib/painter";
import Simulator from "@/lib/simulator";
import styles from "./index.module.css";

function createOffscreenCanvasHandle(
	width: number,
	height: number,
): CanvasHandle<"2d"> {
	const canvas = new OffscreenCanvas(width, height);
	const context = canvas.getContext("2d") ?? todo("2d context를 얻을 수 없음");

	return {
		canvas,
		draw: (pass) => pass(context, width, height),
	};
}

/**
 * 큐의 방향정보와 수구, 목적구 정보 분리
 * @param cuePoints
 * @param ballPoints
 * @returns
 */
function resolveTableState(
	cuePoints: [Vector2<"normalized">, Vector2<"normalized">],
	ballPoints: Vector2<"normalized">[],
): {
	cue: Cue;
	cueBall: Vector2<"normalized"> | null;
	objectBalls: Vector2<"normalized">[];
} {
	let line: Line<"normalized"> = {
		start: cuePoints[0],
		end: cuePoints[1],
	};
	let cueBallCandidate: {
		point: Vector2<"normalized">;
		distance: number;
	} | null = null;
	const objectBalls: Vector2<"normalized">[] = [];

	for (const ballPoint of ballPoints) {
		const distances = cuePoints.map((cuePoint) => dist(ballPoint, cuePoint));
		const cueTipIndex = argmin(distances);
		const minDistance = distances[cueTipIndex];

		if (cueBallCandidate === null || minDistance < cueBallCandidate.distance) {
			if (cueBallCandidate !== null) {
				objectBalls.push(cueBallCandidate.point);
			}

			cueBallCandidate = {
				point: ballPoint,
				distance: minDistance,
			};
			line = {
				start: cuePoints[1 - cueTipIndex],
				end: cuePoints[cueTipIndex],
			};
		} else {
			objectBalls.push(ballPoint);
		}
	}

	return {
		cue: {
			line,
			angle: Math.atan2(line.end.y - line.start.y, line.end.x - line.start.x),
		},
		cueBall: cueBallCandidate?.point ?? null,
		objectBalls,
	};
}

/**
 * 메인 페이지.
 */
function Main() {
	/**
	 * 카메라 프레임 표시하는 video
	 */
	const cameraVideoRef = useRef<HTMLVideoElement>(null);
	/**
	 * 오버레이 Canvas
	 */
	const [createOverlayCanvas, overlayCanvasSpec] = useGPUCanvas();
	/**
	 * Debug Canvas
	 */
	const [
		createDebug2DCanvas,
		createDebugGPUCanvas,
		debugCanvasSpecs,
		clearDebugCanvasSpecs,
	] = useDebugCanvas();

	const hitPointRef = useRef<Vector2<"unit">>({ x: 0, y: 0 });
	const hitPowerRef = useRef(0.5);

	/**
	 * overlay 활성화 여부
	 */
	const [isOverlayEnabled, setIsOverlayEnabled] = useState(false);
	const [isControlUiHidden, setIsControlUiHidden] = useState(false);

	const loop = useEffectEvent(
		async (
			source: HTMLVideoElement,
			cuebit: Cuebit,
			simulator: Simulator,
			trajectoryDrawerCanvas: CanvasHandle<"2d">,
			overlayCanvas: CanvasHandle<"webgpu">,
			trajectoryPainter: TrajectoryPainter,
			textureTransformer: TextureTransformer,
			resizedFrameDebugCanvas: CanvasHandle<"webgpu">,
			tableMaskDebugCanvas: CanvasHandle<"webgpu">,
			cueMaskDebugCanvas: CanvasHandle<"webgpu">,
			detectionDebugCanvas: CanvasHandle<"2d">,
			normalizedTableDebugCanvas: CanvasHandle<"2d">,
			trajectoryDebugCanvas: CanvasHandle<"2d">,
		) => {
			if (!isOverlayEnabled) {
				return;
			}

			const result = await measure(
				() => cuebit.process(source),
				"Process Frame",
			);

			const bufferIndex = cuebit.getCurrentBufferIndex();
			const buffer = cuebit.getBuffer(bufferIndex);

			drawTexture(resizedFrameDebugCanvas, buffer.resizedFrameTexture);
			drawTexture(tableMaskDebugCanvas, buffer.tableMaskFrameTexture);
			drawTexture(cueMaskDebugCanvas, buffer.cueMaskFrameTexture);

			detectionDebugCanvas.draw((context, width, height) => {
				const protoToCanvasX =
					width / onnx.segementation.output.fetchs.protos.width;
				const protoToCanvasY =
					height / onnx.segementation.output.fetchs.protos.height;
				const feedToCanvasX =
					width / onnx.segementation.input.feeds.image.width;
				const feedToCanvasY =
					height / onnx.segementation.input.feeds.image.height;

				context.clearRect(0, 0, width, height);

				if (result.table) {
					context.strokeStyle = "blue";
					context.lineWidth = width * 0.002;

					context.beginPath();

					context.fillText(
						"table",
						((result.table.approximation.mask.detection.bbox.lt.x +
							result.table.approximation.mask.detection.bbox.rb.x) /
							2) *
							feedToCanvasX,
						result.table.approximation.mask.detection.bbox.lt.y * feedToCanvasY,
					);

					context.rect(
						result.table.approximation.mask.detection.bbox.lt.x * feedToCanvasX,
						result.table.approximation.mask.detection.bbox.lt.y * feedToCanvasY,
						(result.table.approximation.mask.detection.bbox.rb.x -
							result.table.approximation.mask.detection.bbox.lt.x) *
							feedToCanvasX,
						(result.table.approximation.mask.detection.bbox.rb.y -
							result.table.approximation.mask.detection.bbox.lt.y) *
							feedToCanvasY,
					);
					context.stroke();

					context.strokeStyle = "rgb(0, 255, 255, 0.8)";
					context.lineWidth = width * 0.01;
					for (const point of result.table.approximation.lines) {
						context.beginPath();
						context.moveTo(
							point.start.x * protoToCanvasX,
							point.start.y * protoToCanvasY,
						);
						context.lineTo(
							point.end.x * protoToCanvasX,
							point.end.y * protoToCanvasY,
						);

						context.stroke();
					}

					if (result.table.transform) {
						context.strokeStyle = "red";
						context.lineWidth = width * 0.005;
						context.font = `${width * 0.02}px Arial`;
						context.fillStyle = "red";
						context.textAlign = "center";
						context.textBaseline = "bottom";

						context.beginPath();

						const points = [
							result.table.transform.quad.points.topLeft,
							result.table.transform.quad.points.bottomLeft,
							result.table.transform.quad.points.bottomRight,
							result.table.transform.quad.points.topRight,
						];

						for (let i = 0; i < 4; i++) {
							const point = points[i];
							context.fillText(
								`${i}`,
								point.x * protoToCanvasX,
								point.y * protoToCanvasY,
							);
							context.moveTo(
								point.x * protoToCanvasX,
								point.y * protoToCanvasY,
							);
							const nextPoint = points[(i + 1) % 4];
							context.lineTo(
								nextPoint.x * protoToCanvasX,
								nextPoint.y * protoToCanvasY,
							);
						}
						context.stroke();
					}
				}

				if (result.cue) {
					context.strokeStyle = "blue";
					context.lineWidth = width * 0.002;

					context.beginPath();

					context.fillText(
						"cue",
						((result.cue.approximation.mask.detection.bbox.lt.x +
							result.cue.approximation.mask.detection.bbox.rb.x) /
							2) *
							feedToCanvasX,
						result.cue.approximation.mask.detection.bbox.lt.y * feedToCanvasY,
					);

					context.rect(
						result.cue.approximation.mask.detection.bbox.lt.x * feedToCanvasX,
						result.cue.approximation.mask.detection.bbox.lt.y * feedToCanvasY,
						(result.cue.approximation.mask.detection.bbox.rb.x -
							result.cue.approximation.mask.detection.bbox.lt.x) *
							feedToCanvasX,
						(result.cue.approximation.mask.detection.bbox.rb.y -
							result.cue.approximation.mask.detection.bbox.lt.y) *
							feedToCanvasY,
					);
					context.stroke();

					if (result.cue.approximation.endpoints) {
						context.strokeStyle = "white";
						context.lineWidth = width * 0.002;

						context.beginPath();
						context.moveTo(
							result.cue.approximation.endpoints[0].x * protoToCanvasX,
							result.cue.approximation.endpoints[0].y * protoToCanvasY,
						);
						context.lineTo(
							result.cue.approximation.endpoints[1].x * protoToCanvasX,
							result.cue.approximation.endpoints[1].y * protoToCanvasY,
						);
						context.stroke();
					}

					context.strokeStyle = "blue";
					context.lineWidth = width * 0.002;

					for (const ball of result.ballPoints) {
						context.beginPath();
						context.arc(
							ball.x * protoToCanvasX,
							ball.y * protoToCanvasY,
							width * 0.02,
							0,
							2 * Math.PI,
						);
						context.stroke();
					}
				}
			});

			const tableTransform = result.table?.transform;
			const cuePoints = result.cue?.approximation?.endpoints;
			if (tableTransform && cuePoints) {
				const normalizedBallPoints = withMatScope((track) => {
					const src = track(
						cv.matFromArray(
							result.ballPoints.length,
							1,
							cv.CV_32FC2,
							result.ballPoints.flatMap((p) => [p.x, p.y]),
						),
					);
					const dst = track(new cv.Mat());
					const transform = track(restoreMat(tableTransform.matrix.transform));
					cv.perspectiveTransform(src, dst, transform);
					const transformedPoints: Vector2<"normalized">[] = [];
					for (let i = 0; i < result.ballPoints.length; i++) {
						transformedPoints.push({
							x: dst.data32F[i * 2],
							y: dst.data32F[i * 2 + 1],
						});
					}

					return transformedPoints;
				});

				const normalizedCuePoints = withMatScope((track) => {
					if (!result.cue) {
						return null;
					}

					const src = track(
						cv.matFromArray(2, 1, cv.CV_32FC2, [
							cuePoints[0].x,
							cuePoints[0].y,
							cuePoints[1].x,
							cuePoints[1].y,
						]),
					);
					const dst = track(new cv.Mat());
					const transform = track(restoreMat(tableTransform.matrix.transform));
					cv.perspectiveTransform(src, dst, transform);
					const points: [Vector2<"normalized">, Vector2<"normalized">] = [
						{
							x: dst.data32F[0],
							y: dst.data32F[1],
						},
						{
							x: dst.data32F[2],
							y: dst.data32F[3],
						},
					];

					return points;
				});

				const resolvedState =
					normalizedCuePoints &&
					resolveTableState(normalizedCuePoints, normalizedBallPoints);

				normalizedTableDebugCanvas.draw((context, width, height) => {
					const normalToWidth = width / 2844;
					const normalToHeight = height / 1422;

					context.clearRect(0, 0, width, height);
					context.lineWidth = width * 0.004;
					context.font = `${width * 0.02}px Arial`;
					context.textAlign = "center";
					context.textBaseline = "bottom";

					if (resolvedState?.objectBalls) {
						context.strokeStyle = "red";
						context.fillStyle = "red";
						for (let i = 0; i < resolvedState.objectBalls.length; i++) {
							const point = resolvedState.objectBalls[i];

							context.beginPath();
							context.arc(
								point.x * normalToWidth,
								point.y * normalToHeight,
								hyperparams.ball.radius * normalToWidth * 1000,
								0,
								2 * Math.PI,
							);
							context.stroke();

							context.fillText(
								`ball ${i}`,
								point.x * normalToWidth,
								point.y * normalToHeight -
									hyperparams.ball.radius * normalToHeight * 1000,
							);
						}
					}

					if (resolvedState?.cue && resolvedState.cueBall) {
						context.strokeStyle = "white";
						context.fillStyle = "white";

						context.beginPath();
						context.arc(
							resolvedState.cueBall.x * normalToWidth,
							resolvedState.cueBall.y * normalToHeight,
							hyperparams.ball.radius * normalToWidth * 1000,
							0,
							2 * Math.PI,
						);
						context.stroke();

						const to: Vector2<"normalized"> = {
							x:
								(resolvedState.cueBall.x +
									hitPowerRef.current *
										height *
										Math.cos(resolvedState.cue.angle)) *
								normalToWidth,
							y:
								(resolvedState.cueBall.y +
									hitPowerRef.current *
										height *
										Math.sin(resolvedState.cue.angle)) *
								normalToHeight,
						};

						const headLength = width * 0.02;
						context.lineWidth = width * 0.002;
						context.beginPath();
						context.moveTo(
							resolvedState.cueBall.x * normalToWidth,
							resolvedState.cueBall.y * normalToHeight,
						);
						context.lineTo(to.x, to.y);
						context.stroke();

						context.beginPath();
						context.moveTo(to.x, to.y);
						context.lineTo(
							to.x -
								headLength * Math.cos(resolvedState.cue.angle - Math.PI / 6),
							to.y -
								headLength * Math.sin(resolvedState.cue.angle - Math.PI / 6),
						);
						context.moveTo(to.x, to.y);
						context.lineTo(
							to.x -
								headLength * Math.cos(resolvedState.cue.angle + Math.PI / 6),
							to.y -
								headLength * Math.sin(resolvedState.cue.angle + Math.PI / 6),
						);
						context.stroke();
					}
				});

				if (resolvedState?.cueBall) {
					const [initialSnapshot, step] = simulator.simulate(
						rerange(resolvedState.cueBall, 2844, 2.844),
						resolvedState.objectBalls.map((p) => rerange(p, 2844, 2.844)),
						resolvedState.cue.angle,
						hitPowerRef.current,
						hitPointRef.current,
					);

					const snapshots = [initialSnapshot];
					measure(() => {
						for (let i = 0; i < 600; i++) {
							const snapshot = step();
							snapshots.push(snapshot);
						}
					}, "Simulate Trajectory");

					trajectoryPainter.drawTrajectories(trajectoryDebugCanvas, snapshots);
					trajectoryPainter.drawTrajectories(trajectoryDrawerCanvas, snapshots);
					textureTransformer.drawTransformed(
						trajectoryDrawerCanvas,
						overlayCanvas,
						tableTransform.matrix.inverseTransform,
					);
				}
			}
		},
	);

	useEffect(() => {
		if (!cameraVideoRef.current) {
			return;
		}
		const video = cameraVideoRef.current;

		// 비동기 작업을 중단하기 위한 AbortController
		const ac = new AbortController();
		let stream: MediaStream | null = null;
		const textureTransformer = new TextureTransformer(device, 2844, 1422);
		const stopStream = () => {
			stream?.getTracks().forEach((track) => {
				track.stop();
			});
		};

		(async () => {
			try {
				const guard = <T,>(p: Promise<T>) =>
					p.then((v) => {
						ac.signal.throwIfAborted();
						return v;
					});

				// 카메라 스트림 가져오기
				stream = await guard(
					navigator.mediaDevices.getUserMedia({
						// 오디오 스트림은 사용하지 않음
						audio: false,
						video: {
							width: 1000,
							height: 1000,
							facingMode: {
								// 후면 카메라 사용
								ideal: "environment",
							},
						},
					}),
				);

				video.srcObject = stream;
				await video.play();
				const frameInfo = await guard(
					new Promise<FrameInfo>((resolve) => {
						if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
							resolve({ width: video.videoWidth, height: video.videoHeight });
						} else {
							video.addEventListener(
								"loadedmetadata",
								() =>
									resolve({
										width: video.videoWidth,
										height: video.videoHeight,
									}),
								{ once: true },
							);
						}
					}),
				);
				logger.info(
					`카메라 스트림 해상도: ${frameInfo.width}x${frameInfo.height}`,
				);

				const trajectoryDrawerCanvas = createOffscreenCanvasHandle(2844, 1422);

				const overlayCanvas = await guard(
					createOverlayCanvas(device, frameInfo.width, frameInfo.height),
				);

				const cuebit = new Cuebit(device, onnx, frameInfo);
				logger.info("Cuebit 파이프라인 준비됨");

				const simulator = new Simulator();
				logger.info("물리 시뮬레이터 준비됨");

				const resizedFrameDebugCanvas = await guard(
					createDebugGPUCanvas(
						device,
						onnx.segementation.input.feeds.image.width,
						onnx.segementation.input.feeds.image.height,
						{
							width: "100cqw",
							height: "auto",
							aspectRatio: "1 / 1",
						},
						"Resized Frame",
					),
				);
				const tableMaskDebugCanvas = await guard(
					createDebugGPUCanvas(
						device,
						onnx.segementation.output.fetchs.protos.width,
						onnx.segementation.output.fetchs.protos.height,
						{
							width: "100cqw",
							height: "100cqh",
							objectFit: "cover",
							objectPosition: "50% 50%",
							opacity: 0.5,
						},
						"Table Mask",
					),
				);
				const cueMaskDebugCanvas = await guard(
					createDebugGPUCanvas(
						device,
						onnx.segementation.output.fetchs.protos.width,
						onnx.segementation.output.fetchs.protos.height,
						{
							width: "100cqw",
							height: "100cqh",
							objectFit: "cover",
							objectPosition: "50% 50%",
							opacity: 0.8,
						},
						"Cue Mask",
					),
				);
				const detectionDebugCanvas = await guard(
					createDebug2DCanvas(
						frameInfo.width,
						frameInfo.height,
						{
							width: "100cqw",
							height: "100cqh",
							objectFit: "cover",
							objectPosition: "50% 50%",
						},
						"Detection Result",
					),
				);
				const normalizedTableDebugCanvas = await guard(
					createDebug2DCanvas(
						2844,
						1422,
						{
							width: "100cqw",
							height: "auto",
							aspectRatio: "2 / 1",
						},
						"Normalized Detection Result",
					),
				);

				const trajectoryDebugCanvas = await guard(
					createDebug2DCanvas(
						2844,
						1422,
						{
							width: "100cqw",
							height: "auto",
							aspectRatio: "2 / 1",
						},
						"Trajectory",
					),
				);

				const trajectoryPainter = new TrajectoryPainter(2844, 1422);

				let busy = false;
				const tick = () => {
					if (ac.signal.aborted) return;
					if (!busy) {
						busy = true;
						loop(
							video,
							cuebit,
							simulator,
							trajectoryDrawerCanvas,
							overlayCanvas,
							trajectoryPainter,
							textureTransformer,
							resizedFrameDebugCanvas,
							tableMaskDebugCanvas,
							cueMaskDebugCanvas,
							detectionDebugCanvas,
							normalizedTableDebugCanvas,
							trajectoryDebugCanvas,
						).finally(() => {
							busy = false;
						});
					}
					video.requestVideoFrameCallback(tick);
				};
				video.requestVideoFrameCallback(tick);
			} catch (error) {
				if (error instanceof Error && error.name === "AbortError") {
					logger.info("초기화 과정이 중단됨");
				} else {
					logger.error(
						{ err: error },
						"Main page initialization or frame loop failed",
					);
				}
			}
		})();

		return () => {
			ac.abort();
			stopStream();
			textureTransformer[Symbol.dispose]();
			clearDebugCanvasSpecs();
		};
	}, [
		createOverlayCanvas,
		createDebug2DCanvas,
		createDebugGPUCanvas,
		clearDebugCanvasSpecs,
	]);

	return (
		<div
			className={styles.container}
			style={{
				width: "100vw",
				height: "100vh",
				containerType: "size",
			}}
		>
			{/* 카메라 프레임 */}
			<div
				style={{
					position: "absolute",
					width: "100cqw",
					height: "100cwh",
				}}
			>
				<video
					ref={cameraVideoRef}
					playsInline
					muted
					autoPlay
					style={{
						width: "100cqw",
						height: "100cqh",
						objectFit: "cover",
						objectPosition: "50% 50%",
					}}
				/>
			</div>

			{/* 오버레이 */}
			{/* NOTE: AR 비활성화 후 디버깅을 위해 꺼도 궤적 남아있도록 한 상태임. 나중에 UX개선을 한다면 isOverlayEnabled 을 조건에 추가해야함 */}
			{overlayCanvasSpec && (
				<div
					style={{
						position: "absolute",
						width: "100cqw",
						height: "100cwh",
					}}
				>
					<canvas
						ref={(element) => {
							if (element) {
								overlayCanvasSpec.onMount(element);
							}
						}}
						width={overlayCanvasSpec.width}
						height={overlayCanvasSpec.height}
						style={{
							width: "100cqw",
							height: "100cqh",
							objectFit: "cover",
							objectPosition: "50% 50%",
						}}
					/>
				</div>
			)}

			{/* 디버깅 */}
			<div
				style={{
					width: "100cqw",
					height: "100cqh",
					overflow: "scroll",
					position: "absolute",
					display: "flex",
					alignItems: "center",
					justifyContent: "flex-start",
					scrollSnapType: "x mandatory",
				}}
			>
				<div
					style={{
						width: "100vw",
						height: "auto",
						display: "flex",
						alignItems: "center",
						justifyContent: "center",
						flexShrink: 0,
						scrollSnapAlign: "center",
					}}
				></div>

				{debugCanvasSpecs.map((spec) => (
					<div
						key={spec.id}
						style={{
							width: "100cqw",
							height: "100cqh",
							position: "relative",
							display: "flex",
							alignItems: "center",
							justifyContent: "center",
							flexShrink: 0,
						}}
					>
						<canvas
							ref={(element) => {
								if (element) {
									spec.onMount(element);
								}
							}}
							width={spec.width}
							height={spec.height}
							style={{
								...spec.style,
								backdropFilter: "brightness(0.6)",
								scrollSnapAlign: "center",
							}}
						/>
						<span
							style={{
								position: "absolute",
								top: 8,
								right: 8,
								padding: "4px",
								backgroundColor: "rgba(0, 0, 0, 0.5)",
								color: "white",
								fontSize: 12,
								borderRadius: 4,
							}}
						>
							{spec.name}
						</span>
					</div>
				))}
			</div>

			{/* 상단 헤더 */}
			<div className={styles.header}>
				<div>
					<h1 className={styles.title}>
						Cue<span className={styles.titleAccent}>bit</span>
					</h1>
					<p className={styles.subtitle}>Real-time Trajectory</p>
				</div>
				{isOverlayEnabled && (
					<div className={styles.analyzingBadge}>
						<div className={styles.analyzingDot} />
						<span className={styles.analyzingText}>실시간 분석 중...</span>
					</div>
				)}
			</div>

			{/* 하단 컨트롤 패널 */}
			<div
				className={`${styles.controls} ${
					isControlUiHidden ? styles.controlsHidden : ""
				}`}
			>
				<div
					className={
						isOverlayEnabled && !isControlUiHidden
							? ""
							: styles.controlPanelHidden
					}
				>
					<HitControlPanel
						onHitPointChange={(point) => {
							hitPointRef.current = point;
						}}
						onHitPowerChange={(power) => {
							hitPowerRef.current = power;
						}}
					/>
				</div>
				<div className={styles.actionRow}>
					{isOverlayEnabled && (
						<button
							type="button"
							className={styles.controlVisibilityButton}
							onClick={() => setIsControlUiHidden((prev) => !prev)}
						>
							{isControlUiHidden ? "UI \ud45c\uc2dc" : "UI \uc228\uae40"}
						</button>
					)}
					{!isControlUiHidden && (
						<OverlayToggleButton
							enabled={isOverlayEnabled}
							onClick={() => {
								if (isOverlayEnabled) {
									setIsControlUiHidden(false);
								}
								setIsOverlayEnabled((prev) => !prev);
							}}
						/>
					)}
				</div>
			</div>

			{/* 개발용 로그 패널 (개발 환경에서만 표시) */}
			{/* <DevLog /> */}
		</div>
	);
}

export default Main;
