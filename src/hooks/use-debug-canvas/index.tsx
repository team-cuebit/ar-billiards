import {
	type CSSProperties,
	type RefCallback,
	useCallback,
	useRef,
	useState,
} from "react";
import { todo } from "@/common";

export type DebugCanvasSpec = {
	id: number;
	width: number;
	height: number;
	style: CSSProperties;
	onMount: RefCallback<HTMLCanvasElement>;
	name?: string;
};

function useDebugCanvas() {
	const id = useRef(0);
	const [specs, setSpecs] = useState<DebugCanvasSpec[]>([]);

	const create2DCanvas = useCallback(
		(
			width: number,
			height: number,
			style: CSSProperties,
			name?: string,
		): Promise<CanvasHandle<"2d">> => {
			return new Promise((resolve) => {
				setSpecs((prev) => [
					...prev,
					{
						id: id.current++,
						width,
						height,
						style,
						onMount: (canvas) => {
							if (!canvas) {
								return;
							}

							const context =
								canvas.getContext("2d") ?? todo(`context를 얻을 수 없음`);

							resolve({
								canvas,
								draw: (pass) => pass(context, width, height),
							});
						},
						name,
					},
				]);
			});
		},
		[],
	);

	const createGPUCanvas = useCallback(
		(
			device: GPUDevice,
			width: number,
			height: number,
			style: CSSProperties,
			name?: string,
		): Promise<CanvasHandle<"webgpu">> => {
			return new Promise((resolve) => {
				setSpecs((prev) => [
					...prev,
					{
						id: id.current++,
						width,
						height,
						style,
						onMount: (canvas) => {
							if (!canvas) {
								return;
							}

							const context =
								canvas.getContext("webgpu") ?? todo(`context를 얻을 수 없음`);

							context.configure({
								device,
								format: "rgba8unorm",
								usage:
									GPUTextureUsage.COPY_SRC |
									GPUTextureUsage.COPY_DST |
									GPUTextureUsage.RENDER_ATTACHMENT,
							});

							resolve({
								canvas,
								draw: (pass) => pass(device, context, width, height),
							});
						},
						name,
					},
				]);
			});
		},
		[],
	);

	const clear = useCallback(() => {
		setSpecs([]);
	}, []);

	return [create2DCanvas, createGPUCanvas, specs, clear] as const;
}
export default useDebugCanvas;
