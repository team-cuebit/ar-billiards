import { useCallback, useState } from "react";
import { todo } from "@/common";

function useGPUCanvas() {
	const [spec, setSpec] = useState<CanvasSpec | null>(null);

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
						if (!canvas) {
							return;
						}

						const context =
							canvas.getContext("webgpu") ??
							todo("webgpu context를 얻을 수 없음");

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
