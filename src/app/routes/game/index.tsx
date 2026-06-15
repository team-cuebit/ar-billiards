import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useRef } from "react";
import type * as THREE from "three";
import {
	CanvasTexture,
	Quaternion,
	RepeatWrapping,
	Vector2 as ThreeVector2,
	Vector3,
} from "three";
import { todo } from "@/common";
import HitControlPanel from "@/components/hit-params-panel";
import hyperparams from "@/config/hyperparams";
import Simulator from "@/lib/simulator";
import { styles } from "./index.css";

const TABLE_WIDTH = 2.844;
const TABLE_HEIGHT = 1.422;

const TABLE_CENTER: [number, number, number] = [
	TABLE_WIDTH / 2,
	0,
	TABLE_HEIGHT / 2,
];

const RAIL_HEIGHT = hyperparams.ball.radius * 0.7;
const RAIL_THICKNESS = 0.05;

// 0번은 큐볼(흰색), 이후는 오브젝트 볼
const BALL_COLORS = [
	"#ffffff",
	"#e53935",
	"#fdd835",
	"#fb8c00",
	"#43a047",
	"#6d4c41",
	"#00acc1",
	"#d81b60",
];

// 큐대 (조준 상태에서만 표시)
const CUE_LENGTH = 1.45;
const CUE_TIP_GAP = 0.015;
const Z_AXIS = new Vector3(0, 0, 1);
const cueDir = new Vector3();
const cueQuat = new Quaternion();

// 회전(스핀)을 눈으로 확인하기 위한 마커 위치 (구 표면의 ±X, ±Y, ±Z 6면)
const MARKER_OFFSET = 0.82;
const MARKER_POSITIONS: Array<[number, number, number]> = [
	[MARKER_OFFSET, 0, 0],
	[-MARKER_OFFSET, 0, 0],
	[0, MARKER_OFFSET, 0],
	[0, -MARKER_OFFSET, 0],
	[0, 0, MARKER_OFFSET],
	[0, 0, -MARKER_OFFSET],
];

function SnapshotRenderer({
	snapshotRef,
	stepRef,
	ballCount,
}: {
	snapshotRef: React.RefObject<TableSnapshot | null>;
	stepRef: React.RefObject<StepFn | null>;
	ballCount: number;
}) {
	const meshRefs = useRef<(THREE.Mesh | null)[]>([]);

	useFrame(() => {
		if (stepRef.current) {
			snapshotRef.current = stepRef.current();
		}

		const snapshot = snapshotRef.current;
		if (!snapshot) {
			return;
		}

		const balls = [snapshot.cueBall, ...snapshot.objectBalls];
		balls.forEach((ball, index) => {
			const mesh = meshRefs.current[index];
			if (!mesh) {
				return;
			}

			mesh.visible = true;
			mesh.position.set(ball.position.x, ball.position.y, ball.position.z);
			mesh.quaternion.set(
				ball.rotation.x,
				ball.rotation.y,
				ball.rotation.z,
				ball.rotation.w,
			);
			mesh.scale.setScalar(ball.radius);
		});

		// 사용하지 않는 메시는 숨김
		for (let i = balls.length; i < meshRefs.current.length; i++) {
			const mesh = meshRefs.current[i];
			if (mesh) {
				mesh.visible = false;
			}
		}
	});

	return (
		<>
			{Array.from({ length: ballCount }).map((_, index) => (
				<mesh
					// biome-ignore lint/suspicious/noArrayIndexKey: 공 개수가 고정이라 인덱스 키가 안전
					key={index}
					ref={(el) => {
						meshRefs.current[index] = el;
					}}
					visible={false}
					castShadow
					receiveShadow
				>
					{/* 반지름 1 구를 만들고 매 프레임 scale로 실제 반지름 반영 */}
					<sphereGeometry args={[1, 32, 32]} />
					<meshStandardMaterial
						color={BALL_COLORS[index % BALL_COLORS.length]}
						roughness={0}
						metalness={0.2}
					/>
					{/* 회전(스핀)을 눈으로 확인하기 위한 마커 6개 (±X, ±Y, ±Z) */}
					{MARKER_POSITIONS.map((position) => (
						<mesh key={position.join(",")} position={position}>
							<sphereGeometry args={[0.22, 16, 16]} />
							<meshStandardMaterial color="#202020" />
						</mesh>
					))}
				</mesh>
			))}
		</>
	);
}

function CueStick({
	snapshotRef,
	stepRef,
	hitAngleRef,
	hitPointRef,
}: {
	snapshotRef: React.RefObject<TableSnapshot | null>;
	stepRef: React.RefObject<StepFn | null>;
	hitAngleRef: React.RefObject<number>;
	hitPointRef: React.RefObject<Vector2<"unit">>;
}) {
	const groupRef = useRef<THREE.Group>(null);

	useFrame(() => {
		const group = groupRef.current;
		const snapshot = snapshotRef.current;
		if (!group || !snapshot) {
			return;
		}

		if (stepRef.current) {
			group.visible = false;
			return;
		}
		group.visible = true;

		const ball = snapshot.cueBall;
		const r = ball.radius;
		const hp = hitPointRef.current;

		const hitAngle = -hitAngleRef.current;
		const dirX = Math.cos(hitAngle);
		const dirZ = Math.sin(hitAngle);
		const perpX = -Math.sin(hitAngle);
		const perpZ = Math.cos(hitAngle);

		// 큐대 팁 = 공 뒤쪽 표면 + 당점(좌우/상하) 오프셋
		group.position.set(
			ball.position.x - dirX * (r + CUE_TIP_GAP) + perpX * r * hp.x,
			ball.position.y + r * hp.y,
			ball.position.z - dirZ * (r + CUE_TIP_GAP) + perpZ * r * hp.x,
		);

		// 큐대는 공에서 멀어지는 방향(-dir)으로 뻗는다 → local +Z 를 정렬
		cueDir.set(-dirX, 0, -dirZ);
		cueQuat.setFromUnitVectors(Z_AXIS, cueDir);
		group.quaternion.copy(cueQuat);
	});

	return (
		<group ref={groupRef} visible={false}>
			{/* 샤프트 */}
			<mesh
				position={[0, 0, CUE_LENGTH / 2 + CUE_TIP_GAP]}
				rotation={[Math.PI / 2, 0, 0]}
				castShadow
			>
				{/* radiusTop=버트(굵음, +Z쪽), radiusBottom=팁(가늘음, 공쪽) */}
				<cylinderGeometry args={[0.014, 0.006, CUE_LENGTH, 16]} />
				<meshStandardMaterial color="#c9a14a" roughness={0.5} />
			</mesh>
			{/* 큐팁 */}
			<mesh position={[0, 0, CUE_TIP_GAP / 2]} rotation={[Math.PI / 2, 0, 0]}>
				<cylinderGeometry args={[0.006, 0.006, CUE_TIP_GAP, 16]} />
				<meshStandardMaterial color="#1b6fb3" roughness={0.6} />
			</mesh>
		</group>
	);
}

type FeltTextures = {
	colorMap: CanvasTexture;
	normalMap: CanvasTexture;
};

type FeltOptions = {
	/** 텍스처 한 변의 픽셀 수 */
	size?: number;
	/** value-noise 격자 수 (작을수록 큰 얼룩) */
	noiseGrid?: number;
	/** UV 반복 횟수 [가로, 세로] */
	repeat?: [number, number];
	/** 천 기본색 (0~255 RGB) */
	baseRgb?: [number, number, number];
	/** 얼룩 밝기 범위 [최소, 변동폭] */
	shade?: [number, number];
	/** 노멀맵 기울기 세기 */
	bump?: number;
};

type PaintFn = (
	data: Uint8ClampedArray,
	offset: number,
	x: number,
	y: number,
) => void;

const smoothstep = (t: number) => t * t * (3 - 2 * t);

/** grid×grid 토러스 위에서 bilinear 보간되는 value noise. 반환값 [0,1]. */
function createValueNoise(grid: number): (u: number, v: number) => number {
	const cell = Float32Array.from({ length: grid * grid }, () => Math.random());
	const at = (gx: number, gy: number) =>
		cell[(((gy % grid) + grid) % grid) * grid + (((gx % grid) + grid) % grid)];

	return (u, v) => {
		const x = u * grid;
		const y = v * grid;
		const x0 = Math.floor(x);
		const y0 = Math.floor(y);
		const sx = smoothstep(x - x0);
		const sy = smoothstep(y - y0);
		const top = at(x0, y0) * (1 - sx) + at(x0 + 1, y0) * sx;
		const bottom = at(x0, y0 + 1) * (1 - sx) + at(x0 + 1, y0 + 1) * sx;
		return top * (1 - sy) + bottom * sy;
	};
}

/** size×size 캔버스를 픽셀 단위로 칠해 반복 텍스처로 감싼다. */
function paintTexture(
	size: number,
	repeat: [number, number],
	paint: PaintFn,
): CanvasTexture {
	const canvas = document.createElement("canvas");
	canvas.width = size;
	canvas.height = size;
	const ctx = canvas.getContext("2d") ?? todo("2D 컨텍스트 미지원 환경");

	const image = ctx.createImageData(size, size);
	for (let y = 0; y < size; y++) {
		for (let x = 0; x < size; x++) {
			paint(image.data, (y * size + x) * 4, x, y);
		}
	}
	ctx.putImageData(image, 0, 0);

	const texture = new CanvasTexture(canvas);
	texture.wrapS = RepeatWrapping;
	texture.wrapT = RepeatWrapping;
	texture.repeat.set(...repeat);
	return texture;
}

/**
 * 펠트 느낌의 컬러맵 + 노멀맵을 만든다. 부드러운 value-noise를 height로 삼아
 * 컬러는 명암 얼룩으로, 노멀은 height의 기울기(요철)로 변환한다.
 */
function createFeltTextures({
	size = 2048,
	noiseGrid = 512,
	repeat = [2, 1],
	baseRgb = [71, 204, 197],
	shade = [0.78, 0.44],
	bump = 0.01,
}: FeltOptions = {}): FeltTextures {
	const noise = createValueNoise(noiseGrid);
	const [baseR, baseG, baseB] = baseRgb;
	const [shadeMin, shadeRange] = shade;

	// height를 한 번만 계산해 두 맵에서 공유
	const height = Float32Array.from({ length: size * size }, (_, i) =>
		noise((i % size) / size, Math.floor(i / size) / size),
	);
	const heightAt = (x: number, y: number) =>
		height[(((y % size) + size) % size) * size + (((x % size) + size) % size)];

	const colorMap = paintTexture(size, repeat, (data, i, x, y) => {
		const shading = shadeMin + heightAt(x, y) * shadeRange;
		data[i] = baseR * shading;
		data[i + 1] = baseG * shading;
		data[i + 2] = baseB * shading;
		data[i + 3] = 255;
	});

	const normalMap = paintTexture(size, repeat, (data, i, x, y) => {
		const dx = (heightAt(x - 1, y) - heightAt(x + 1, y)) * bump;
		const dy = (heightAt(x, y - 1) - heightAt(x, y + 1)) * bump;
		data[i] = 128 + dx * 127;
		data[i + 1] = 128 + dy * 127;
		data[i + 2] = 255;
		data[i + 3] = 255;
	});

	return { colorMap, normalMap };
}

function Table() {
	const railColor = "#007b76";

	const felt = useMemo(() => createFeltTextures(), []);
	const feltNormalScale = useMemo(() => new ThreeVector2(1, 1), []);

	// 언마운트 시 GPU 텍스처 해제
	useEffect(
		() => () => {
			felt.colorMap.dispose();
			felt.normalMap.dispose();
		},
		[felt],
	);

	return (
		<group>
			{/* 베드 */}
			<mesh
				position={[TABLE_WIDTH / 2, -0.025, TABLE_HEIGHT / 2]}
				receiveShadow
			>
				<boxGeometry args={[TABLE_WIDTH, 0.05, TABLE_HEIGHT]} />
				{/* 당구 천(펠트) 느낌: 얼룩 컬러맵 + 요철 노멀맵 + sheen 벨벳 광택 */}
				<meshPhysicalMaterial
					map={felt.colorMap}
					roughness={1}
					metalness={0}
					sheen={1}
					sheenRoughness={0.6}
					sheenColor="#47ccc5"
					normalMap={felt.normalMap}
					normalScale={feltNormalScale}
				/>
			</mesh>

			{/* 쿠션 (좌/우/상/하) */}
			<mesh
				position={[-RAIL_THICKNESS / 2, RAIL_HEIGHT / 2, TABLE_HEIGHT / 2]}
				castShadow
				receiveShadow
			>
				<boxGeometry
					args={[
						RAIL_THICKNESS,
						RAIL_HEIGHT,
						TABLE_HEIGHT + RAIL_THICKNESS * 2,
					]}
				/>
				<meshStandardMaterial color={railColor} roughness={0.8} />
			</mesh>
			<mesh
				position={[
					TABLE_WIDTH + RAIL_THICKNESS / 2,
					RAIL_HEIGHT / 2,
					TABLE_HEIGHT / 2,
				]}
				castShadow
				receiveShadow
			>
				<boxGeometry
					args={[
						RAIL_THICKNESS,
						RAIL_HEIGHT,
						TABLE_HEIGHT + RAIL_THICKNESS * 2,
					]}
				/>
				<meshStandardMaterial color={railColor} roughness={0.8} />
			</mesh>
			<mesh
				position={[TABLE_WIDTH / 2, RAIL_HEIGHT / 2, -RAIL_THICKNESS / 2]}
				castShadow
				receiveShadow
			>
				<boxGeometry args={[TABLE_WIDTH, RAIL_HEIGHT, RAIL_THICKNESS]} />
				<meshStandardMaterial color={railColor} roughness={0.8} />
			</mesh>
			<mesh
				position={[
					TABLE_WIDTH / 2,
					RAIL_HEIGHT / 2,
					TABLE_HEIGHT + RAIL_THICKNESS / 2,
				]}
				castShadow
				receiveShadow
			>
				<boxGeometry args={[TABLE_WIDTH, RAIL_HEIGHT, RAIL_THICKNESS]} />
				<meshStandardMaterial color={railColor} roughness={0.8} />
			</mesh>
		</group>
	);
}

function Game() {
	const simulatorRef = useRef<Simulator>(new Simulator());
	const hitPointRef = useRef<Vector2<"unit">>({ x: 0, y: 0 });
	const hitPowerRef = useRef(0.5);
	const hitAngleRef = useRef(0);

	const cueBallPosRef = useRef<Vector2<"physics">>({
		x: TABLE_WIDTH * 0.25,
		y: TABLE_HEIGHT / 2,
	});
	const objectBallPositionsRef = useRef<Vector2<"physics">[]>([
		{ x: TABLE_WIDTH * 0.7, y: TABLE_HEIGHT / 2 },
		{ x: TABLE_WIDTH * 0.78, y: TABLE_HEIGHT / 2 - 0.06 },
		{ x: TABLE_WIDTH * 0.78, y: TABLE_HEIGHT / 2 + 0.06 },
	]);

	const snapshotRef = useRef<TableSnapshot | null>(null);
	const stepRef = useRef<StepFn | null>(null);

	const ballCount = 1 + objectBallPositionsRef.current.length;

	// 시작 시 공들을 초기 위치에 정적으로 배치(임펄스 0 → 진행 안 함)
	const reset = useCallback(() => {
		const [initial] = simulatorRef.current.simulate(
			cueBallPosRef.current,
			objectBallPositionsRef.current,
			0,
			0,
			{ x: 0, y: 0 },
		);
		snapshotRef.current = initial;
        stepRef.current = null;
	}, []);

	const simulate = useCallback(() => {
		const [initial, step] = simulatorRef.current.simulate(
			cueBallPosRef.current,
			objectBallPositionsRef.current,
			-hitAngleRef.current,
			hitPowerRef.current,
			hitPointRef.current,
		);
		snapshotRef.current = initial;
		stepRef.current = step;
	}, []);

	useEffect(() => {
		reset();
	}, [reset]);

	return (
		<div className={styles.root}>
			<div
				style={{
					width: "100%",
					height: "100%",
					display: "flex",
					justifyContent: "center",
					alignItems: "center",
				}}
			>
				<div
					role="application"
					style={{
						width: "100%",
						height: "100%",
						backgroundColor: "rgba(120, 120, 120, 0.5)",
						position: "relative",
					}}
					onContextMenu={(event) => {
						event.preventDefault();
					}}
				>
					<Canvas
						shadows
						camera={{ position: [TABLE_WIDTH / 2, 2.2, 2.8], fov: 50 }}
					>
						{/* ambient를 낮춰야 요철/sheen 음영이 죽지 않고 보인다 */}
						<ambientLight intensity={0.8} />
						<directionalLight
							position={[TABLE_WIDTH / 2 + 1.5, 6, 4]}
							intensity={1.8}
							castShadow
							shadow-mapSize-width={2048}
							shadow-mapSize-height={2048}
							shadow-camera-left={-4}
							shadow-camera-right={4}
							shadow-camera-top={4}
							shadow-camera-bottom={-4}
							shadow-camera-near={0.1}
							shadow-camera-far={20}
							shadow-bias={0.0}
						/>
						{/* <directionalLight position={[-2, 4, -3]} intensity={0.4} /> */}

						<OrbitControls
							target={TABLE_CENTER}
							enablePan
							maxPolarAngle={Math.PI / 2 - 0.05}
						/>

						<Table />
						<SnapshotRenderer
							snapshotRef={snapshotRef}
							stepRef={stepRef}
							ballCount={ballCount}
						/>
						<CueStick
							snapshotRef={snapshotRef}
							stepRef={stepRef}
							hitAngleRef={hitAngleRef}
							hitPointRef={hitPointRef}
						/>
					</Canvas>
				</div>
			</div>

			<div
				style={{
					position: "absolute",
					bottom: "20px",
					left: "20px",
				}}
			>
				<HitControlPanel
					onHitPointChange={(point) => {
						hitPointRef.current = point;
					}}
					onHitPowerChange={(power) => {
						hitPowerRef.current = power;
					}}
					onHitAngleChange={(angle) => {
						hitAngleRef.current = angle;
					}}
				/>
			</div>
			<div
				style={{
					position: "absolute",
					bottom: "20px",
					right: "20px",
					display: "flex",
					gap: "12px",
					flexDirection: "column",
				}}
			>
				<button onClick={reset} type="button">
					Reset
				</button>
				<button onClick={simulate} type="button">
					Simulate
				</button>
			</div>
		</div>
	);
}

export default Game;
