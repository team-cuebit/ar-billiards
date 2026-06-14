import type { Quaternion, Vector3 } from "@dimforge/rapier3d";

declare module "*.wgsl";

declare global {
	// https://fontsource.org/docs/getting-started/install#4-typescript-configuration
	declare module "*.css";
	declare module "@fontsource/*" {}
	declare module "@fontsource-variable/*" {}

	type ContextMap = {
		"2d": CanvasRenderingContext2D;
		webgpu: GPUCanvasContext;
	};

	type Pass2D = (
		context: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
		width: number,
		height: number,
	) => void;

	type PassWebGPU = (
		device: GPUDevice,
		context: GPUCanvasContext,
		width: number,
		height: number,
	) => void;

	type CanvasSpec = {
		width: number;
		height: number;
		onMount: React.RefCallback<HTMLCanvasElement>;
	};

	type CanvasHandle<T extends keyof ContextMap> = {
		canvas: HTMLCanvasElement | OffscreenCanvas;
		draw: (pass: T extends "2d" ? Pass2D : PassWebGPU) => void;
	};

	/**
	 * OpenCV Mat의 JS 복사본
	 */
	type MatSnapshot = {
		readonly rows: number;
		readonly cols: number;
		readonly type: number;
		readonly data: ArrayBufferLike;
	};

	/**
	 * - **frame**: 카메라로 읽은 프레임 좌표계
	 * - **feed**: ONNX 모델의 입력 좌표계
	 * - **fetch**: ONNX 모델의 출력 좌표계
	 * - **physics**: 물리 시뮬레이터 좌표계
	 * - **unit**: 단위 좌표계
	 */
	type VectorSpace =
		| "frame"
		| "feed"
		| "fetch"
		| "normalized"
		| "physics"
		| "unit";
	type Vector2<S extends VectorSpace> = {
		readonly x: number;
		readonly y: number;
		readonly __space?: S;
	};

	type BoundingBox<S extends VectorSpace> = {
		readonly lt: Vector2<S>;
		readonly rb: Vector2<S>;
	};

	/**
	 * 공의 순간 상태 스냅샷
	 */
	type BallSnapshot = {
		readonly position: Vector3;
		readonly rotation: Quaternion;
		readonly linvel: Vector3;
		readonly angvel: Vector3;
		readonly radius: number;
		readonly collided: boolean;
	};

	/**
	 * 테이블의 순간 상태 스냅샷
	 */
	type TableSnapshot = {
		readonly cueBall: BallSnapshot;
		readonly objectBalls: BallSnapshot[];
	};

	/**
	 * 단일 공의 궤적 정보
	 */
	type BallTrajectory = {
		readonly snapshots: BallSnapshot[];
	};

	type TableApproximation = {
		points:
			| [Vector2<"fetch">, Vector2<"fetch">, Vector2<"fetch">, Vector2<"fetch">]
			| null;
		hulls: Vector2<"fetch">[];
	};

	type Quad<S extends VectorSpace> = {
		readonly points: {
			readonly topLeft: Vector2<S>;
			readonly bottomLeft: Vector2<S>;
			readonly bottomRight: Vector2<S>;
			readonly topRight: Vector2<S>;
		};
	};

	type Line<S extends VectorSpace> = {
		readonly start: Vector2<S>;
		readonly end: Vector2<S>;
	};

	type Cue = {
		readonly line: Line<"normalized">;
		/**
		 * radian
		 */
		readonly angle: number;
	};
}
