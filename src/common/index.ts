import cv from "@techstark/opencv-js";
import logger from "@/lib/logger";

export function todo<T>(message: string): NonNullable<T> {
	throw new Error(`TODO: ${message}`);
}

export function measure<T>(fn: () => Promise<T>, tag?: string): Promise<T>;
export function measure<T>(fn: () => T, tag?: string): T;
export function measure<T>(
	fn: () => T | Promise<T>,
	tag?: string,
): T | Promise<T> {
	const start = performance.now();
	const result = fn();

	if (result instanceof Promise) {
		return result.then((val) => {
			logger.debug(
				`Execution time${tag ? ` (${tag})` : ""}: ${(performance.now() - start).toFixed(2)} ms`,
			);
			return val;
		});
	}

	logger.debug(
		`Execution time${tag ? ` (${tag})` : ""}: ${(performance.now() - start).toFixed(2)} ms`,
	);
	return result;
}

interface ScopedMeasure {
	<R>(fn: () => Promise<R>, tag?: string): Promise<R>;
	<R>(fn: () => R, tag?: string): R;
}

export function measureScope<T>(
	scope: (measure: ScopedMeasure) => T,
	groupTag: string,
): T {
	let accumulated = 0;

	const log = (duration: number, tag?: string) => {
		accumulated += duration;
		logger.debug(
			`[${groupTag}] executed in ${duration.toFixed(2)} ms (accumulated: ${accumulated.toFixed(2)} ms) ${tag ?? "Task"} `,
		);
	};

	const measure = (<R>(task: () => R | Promise<R>, tag?: string) => {
		const start = performance.now();
		const result = task();

		if (result instanceof Promise) {
			return result.then((val) => {
				log(performance.now() - start, tag);
				return val;
			});
		}

		log(performance.now() - start, tag);
		return result;
	}) as ScopedMeasure;

	return scope(measure);
}

/**
 * 16배수 정렬
 */
export function alignTo16(size: number): number {
	return Math.ceil(size / 16) * 16;
}

/**
 * Mat의 수명 관리를 도와주는 유틸리티 함수
 * @param fn
 * @returns
 */
export function withMatScope<T>(
	fn: (track: <M extends { delete(): void }>(m: M) => M) => T,
): T {
	const allocated: { delete(): void }[] = [];
	const track = <M extends { delete(): void }>(m: M) => {
		allocated.push(m);
		return m;
	};
	try {
		return fn(track);
	} finally {
		for (const m of allocated) m.delete();
	}
}

/**
 * Mat을 JS에서 관리할 수 있도록 직렬화
 * @param mat
 * @returns
 */
export function snapshotMat(mat: cv.Mat): MatSnapshot {
	// mat.data는 항상 Uint8Array view (byte-level)
	// 하지만 byteLength가 element 수가 아니라 실제 바이트 수
	const bytes = new Uint8Array(
		mat.data.buffer,
		mat.data.byteOffset,
		mat.rows * mat.cols * mat.elemSize(),
	);
	return {
		rows: mat.rows,
		cols: mat.cols,
		type: mat.type(),
		data: bytes.slice().buffer,
	};
}

/**
 * snapshotMat으로 직렬화된 Mat을 복원
 * @param snap
 * @returns
 */
export function restoreMat(snap: MatSnapshot): cv.Mat {
	const mat = new cv.Mat(snap.rows, snap.cols, snap.type);
	mat.data.set(new Uint8Array(snap.data));
	return mat;
}

export function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export function rerange<From extends VectorSpace, To extends VectorSpace>(
	vector: Vector2<From>,
	from: number,
	to: number,
): Vector2<To> {
	const scale = to / from;
	return {
		x: vector.x * scale,
		y: vector.y * scale,
	};
}

export function exportMatToPNG(mat: cv.Mat, fileName = "output.png") {
	const canvas = document.createElement("canvas");

	cv.imshow(canvas, mat);

	const link = document.createElement("a");
	link.href = canvas.toDataURL("image/png");
	link.download = fileName;
	link.click();
}

export async function exportCanvasToPNG(
	canvas: HTMLCanvasElement | OffscreenCanvas,
	fileName = "output.png",
) {
	const link = document.createElement("a");
	link.href =
		canvas instanceof HTMLCanvasElement
			? canvas.toDataURL("image/png")
			: URL.createObjectURL(await canvas.convertToBlob());
	link.download = fileName;
	link.click();
}

export async function exportGPUTextureToPNG(
	device: GPUDevice,
	texture: GPUTexture,
	fileName = "output.png",
) {
	// rgba8
	const bytesPerPixel = 4;
	const unalignedBytesPerRow = texture.width * bytesPerPixel;
	// 256바이트 정렬
	const bytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;
	const bufferSize = bytesPerRow * texture.height;

	// 데이터를 복사받을 CPU 맵핑용 버퍼
	const outputBuffer = device.createBuffer({
		size: bufferSize,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	const commandEncoder = device.createCommandEncoder();
	commandEncoder.copyTextureToBuffer(
		{
			texture,
		},
		{
			buffer: outputBuffer,
			bytesPerRow,
			rowsPerImage: texture.height,
		},
		{
			width: texture.width,
			height: texture.height,
			depthOrArrayLayers: 1,
		},
	);
	device.queue.submit([commandEncoder.finish()]);

	// GPU 작업 완료 대기 및 버퍼 맵핑
	await outputBuffer.mapAsync(GPUMapMode.READ);
	const arrayBuffer = outputBuffer.getMappedRange();
	const rawData = new Uint8Array(arrayBuffer);

	// Canvas 생성 및 패딩 제거 후 데이터 복사
	const canvas = document.createElement("canvas");
	canvas.width = texture.width;
	canvas.height = texture.height;
	const context = canvas.getContext("2d") ?? todo("2d context not supported");
	const imageData = context.createImageData(texture.width, texture.height);

	// 패딩 제거
	for (let y = 0; y < texture.height; y++) {
		const srcOffset = y * bytesPerRow;
		const destOffset = y * texture.width * bytesPerPixel;
		imageData.data.set(
			rawData.subarray(srcOffset, srcOffset + unalignedBytesPerRow),
			destOffset,
		);
	}
	context.putImageData(imageData, 0, 0);

	// 사용한 버퍼 언맵핑 (메모리 해제)
	outputBuffer.unmap();
	outputBuffer.destroy();

	const dataURL = canvas.toDataURL("image/png");
	const link = document.createElement("a");
	link.download = fileName;
	link.href = dataURL;
	link.click();
}

export function argmin<T>(array: T[]): number {
	let minIndex = 0;
	let minValue = array[0];
	for (let i = 1; i < array.length; i++) {
		if (array[i] < minValue) {
			minValue = array[i];
			minIndex = i;
		}
	}
	return minIndex;
}

export function dist<S extends VectorSpace>(
	a: Vector2<S>,
	b: Vector2<S>,
): number {
	return Math.hypot(a.x - b.x, a.y - b.y);
}
