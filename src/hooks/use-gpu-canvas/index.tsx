import { useCallback, useRef, useState } from "react";
import { todo } from "@/common";

function useGPUCanvas() {
	const [spec, setSpec] = useState<CanvasSpec | null>(null);
	// 인라인 ref 콜백은 매 렌더마다 onMount를 다시 호출함.
	// configure()를 다시 호출하면 캔버스가 초기화되어 그려둔 잔상이 사라지므로,
	// 동일한 캔버스 요소에 대해서는 한 번만 configure 하도록 추적함.
	const configuredCanvasRef = useRef<HTMLCanvasElement | null>(null);

	const createCanvas = useCallback(
		(
			device: GPUDevice,
			width: number,
			height: number,
		): Promise<CanvasHandle<"webgpu">> => {
			return new Promise((resolve) => {
				setSpec({
					width,
					height,
					onMount: (canvas) => {
						const context =
							canvas.getContext("webgpu") ??
							todo("webgpu context를 얻을 수 없음");

						if (configuredCanvasRef.current !== canvas) {
							context.configure({
								device,
								format: "rgba8unorm",
								// AR 오버레이의 빈 배경을 투명하게 만들기 위한 WebGPU 설정
								alphaMode: "premultiplied",
								usage:
									GPUTextureUsage.COPY_SRC |
									GPUTextureUsage.COPY_DST |
									GPUTextureUsage.RENDER_ATTACHMENT,
							});
							configuredCanvasRef.current = canvas;
						}

						resolve({
							canvas,
							draw: (pass) => pass(device, context, width, height),
						});
					},
				});
			});
		},
		[],
	);

	return [createCanvas, spec] as const;
}

export default useGPUCanvas;
