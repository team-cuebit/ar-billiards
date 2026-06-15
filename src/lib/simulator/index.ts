import RAPIER, { Vector3 } from "@dimforge/rapier3d";
import { RawIntegrationParameters } from "@dimforge/rapier3d/rapier_wasm3d";
import hyperparams from "@/config/hyperparams";
import logger from "@/lib/logger";

type SimulationConfig = {
	table: {
		width: number;
		height: number;
	};
	ball: {
		maxCount: number;
		radius: number;
	};
	physics: {
		timeStep: number;
		slidingFriction: number;
		rollingFriction: number;
		spinningFriction: number;
	};
};

type CubitObject = {
	rigidbody: RAPIER.RigidBody;
	collider: RAPIER.Collider;
};

class Simulator {
	private config: SimulationConfig;
	private world: RAPIER.World;
	private eventQueue: RAPIER.EventQueue;
	// @ts-expect-error - table 아직 쓸 일이 없음
	private table: CubitObject[];
	private cueBall: CubitObject;
	private objectBalls: CubitObject[];

	public constructor(
		config: SimulationConfig = {
			table: {
				width: 2.844,
				height: 1.422,
			},
			ball: {
				maxCount: 10,
				radius: hyperparams.ball.radius,
			},
			physics: {
				timeStep: 1 / 240,
				slidingFriction: 0.2,
				rollingFriction: 0.01,
				spinningFriction: 0.04,
			},
		},
	) {
		this.config = config;
		const integrationParameters = new RawIntegrationParameters();
		integrationParameters.dt = config.physics.timeStep;
		integrationParameters.numSolverIterations = 64;
		this.world = new RAPIER.World(
			{ x: 0, y: -9.81, z: 0 },
			integrationParameters,
		);
		this.eventQueue = new RAPIER.EventQueue(true);
		this.table = [
			// ground
			this.createWall(
				new RAPIER.Vector3(config.table.width / 2, -1, config.table.height / 2),
				new RAPIER.Vector3(config.table.width / 2, 1, config.table.height / 2),
			),
			// left
			this.createWall(
				new RAPIER.Vector3(-1, 0, config.table.height / 2),
				new RAPIER.Vector3(1, 5, config.table.height / 2),
			),
			// right
			this.createWall(
				new RAPIER.Vector3(config.table.width + 1, 0, config.table.height / 2),
				new RAPIER.Vector3(1, 5, config.table.height / 2),
			),
			// top
			this.createWall(
				new RAPIER.Vector3(config.table.width / 2, 0, -1),
				new RAPIER.Vector3(config.table.width / 2, 5, 1),
			),
			// bottom
			this.createWall(
				new RAPIER.Vector3(config.table.width / 2, 0, config.table.height + 1),
				new RAPIER.Vector3(config.table.width / 2, 5, 1),
			),
		];
		this.cueBall = this.createBall(config.ball.radius);
		this.objectBalls = Array.from({ length: config.ball.maxCount - 1 }, () =>
			this.createBall(config.ball.radius),
		);
	}

	private createWall(position: RAPIER.Vector3, halfSize: RAPIER.Vector3) {
		const rigidbody = this.world.createRigidBody(
			RAPIER.RigidBodyDesc.fixed().setTranslation(
				position.x,
				position.y,
				position.z,
			),
		);
		const collider = this.world.createCollider(
			RAPIER.ColliderDesc.cuboid(halfSize.x, halfSize.y, halfSize.z)
				.setRestitution(0.5)
				.setFriction(0.4),
			rigidbody,
		);

		return { rigidbody, collider };
	}

	private createBall(radius: number) {
		const rigidbody = this.world.createRigidBody(
			RAPIER.RigidBodyDesc.dynamic()
				.setCcdEnabled(true)
				.setLinearDamping(0.2)
				.setAngularDamping(2)
				.setTranslation(0, 0, 0)
				.setCanSleep(false),
		);

		const collider = this.world.createCollider(
			RAPIER.ColliderDesc.ball(radius)
				.setRestitution(0.95)
				.setFriction(this.config.physics.slidingFriction)
				.setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Min)
				.setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max)
				.setDensity(1700)
				.setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS),
			rigidbody,
		);

		return { rigidbody, collider };
	}

	private applyRollingResistance(ball: CubitObject) {
		const v = ball.rigidbody.linvel();
		const ω = ball.rigidbody.angvel();
		const r = this.config.ball.radius;
		const m = ball.rigidbody.mass();
		const g = 9.81;
		const dt = this.config.physics.timeStep;

		// 접촉점에서 미끄럼 속도: v_contact = (v.x + r·ω.z, 0, v.z - r·ω.x)
		const slipX = v.x + ω.z * r;
		const slipZ = v.z - ω.x * r;
		const slipSpeed = Math.hypot(slipX, slipZ);
		const speed = Math.hypot(v.x, v.z);

		let vx = v.x;
		let vz = v.z;
		let ωx = ω.x;
		let ωz = ω.z;

		// 1. Rolling resistance: 순수 굴림 상태에서 v와 굴림 ω를 같은 비율로 감속
		//    → 굴림 조건(v = ω × -R ŷ) 유지하면서 정지
		if (slipSpeed < 0.02 && speed > 1e-4) {
			const Δv = Math.min(this.config.physics.rollingFriction * g * dt, speed);
			const factor = (speed - Δv) / speed;
			vx *= factor;
			vz *= factor;
			ωx *= factor;
			ωz *= factor;
		}

		// 2. Spinning friction: y축 회전 감쇠. 한 step 감속량이 |ω.y|보다 크면 0으로 클램프
		//    → sign() overshoot으로 인한 부호 진동 방지
		const I_y = (2 / 5) * m * r * r;
		const Δωy = ((this.config.physics.spinningFriction * m * g * r) / I_y) * dt;
		const ωy = Math.abs(ω.y) <= Δωy ? 0 : ω.y - Math.sign(ω.y) * Δωy;

		ball.rigidbody.setLinvel(new Vector3(vx, v.y, vz), true);
		ball.rigidbody.setAngvel(new Vector3(ωx, ωy, ωz), true);
	}

	public simulate(
		cueBallPosition: Vector2<"physics">,
		objectBallPositions: Vector2<"physics">[],
		angle: number,
		power: number,
		hitPoint: Vector2<"unit">,
	): [TableSnapshot, StepFn] {
		if (objectBallPositions.length > this.objectBalls.length) {
			logger.warn("Too many balls");
		}

		this.cueBall.rigidbody.setTranslation(
			new Vector3(
				cueBallPosition.x,
				this.config.ball.radius,
				cueBallPosition.y,
			),
			true,
		);
		this.cueBall.rigidbody.setLinvel(new Vector3(0, 0, 0), true);
		this.cueBall.rigidbody.setAngvel(new Vector3(0, 0, 0), true);
		this.cueBall.rigidbody.setRotation(new RAPIER.Quaternion(0, 0, 0, 1), true);
		this.cueBall.rigidbody.resetForces(true);
		this.cueBall.rigidbody.resetTorques(true);
		this.objectBalls.forEach((ball, i) => {
			if (i < objectBallPositions.length) {
				const position = objectBallPositions[i];
				ball.rigidbody.setTranslation(
					new Vector3(position.x, this.config.ball.radius, position.y),
					true,
				);
				ball.rigidbody.setLinvel(new Vector3(0, 0, 0), true);
				ball.rigidbody.setAngvel(new Vector3(0, 0, 0), true);
				ball.rigidbody.setRotation(new RAPIER.Quaternion(0, 0, 0, 1), true);
				ball.rigidbody.resetForces(true);
				ball.rigidbody.resetTorques(true);
			} else {
				// 테이블 아래로 이동시켜서 시뮬레이션에 영향 안끼치도록
				ball.rigidbody.setTranslation(new Vector3(0, -100, 0), false);
			}
		});

		const initialSnapshot: TableSnapshot = {
			cueBall: {
				position: this.cueBall.rigidbody.translation(),
				rotation: this.cueBall.rigidbody.rotation(),
				linvel: this.cueBall.rigidbody.linvel(),
				angvel: this.cueBall.rigidbody.angvel(),
				radius: this.config.ball.radius,
				collided: false,
			},
			objectBalls: this.objectBalls
				.slice(0, objectBallPositions.length)
				.map((ball) => ({
					position: ball.rigidbody.translation(),
					rotation: ball.rigidbody.rotation(),
					linvel: ball.rigidbody.linvel(),
					angvel: ball.rigidbody.angvel(),
					radius: this.config.ball.radius,
					collided: false,
				})),
		};

		const ballCenter = this.cueBall.rigidbody.translation();

		// 임펄스 방향
		const dirX = Math.cos(angle);
		const dirZ = Math.sin(angle);

		// 수평 평면상 임펄스에 수직인 방향
		const perpX = -Math.sin(angle);
		const perpZ = Math.cos(angle);

		const contactPoint = new Vector3(
			ballCenter.x + perpX * this.config.ball.radius * hitPoint.x,
			ballCenter.y + this.config.ball.radius * hitPoint.y,
			ballCenter.z + perpZ * this.config.ball.radius * hitPoint.x,
		);

		logger.debug(
			`hitPoint: (${hitPoint.x.toFixed(2)}, ${hitPoint.y.toFixed(2)}), contactPoint: (${contactPoint.x.toFixed(2)}, ${contactPoint.y.toFixed(2)}, ${contactPoint.z.toFixed(2)})`,
		);
		this.cueBall.rigidbody.applyImpulseAtPoint(
			new Vector3(power * dirX, 0, power * dirZ),
			contactPoint,
			true,
		);

		// 시뮬레이션에 참여하는 공만 substep 속도 계산에 사용한다.
		// (비활성 공은 y=-100에서 중력으로 무한 낙하해 속도가 계속 커지므로 제외)
		const activeBalls = [
			this.cueBall,
			...this.objectBalls.slice(0, objectBallPositions.length),
		];

		return [
			initialSnapshot,
			() => {
				// 세게 칠수록 한 스텝(1/240s) 변위가 공 반지름을 넘어가는데,
				// 그러면 공이 다른 공/쿠션을 깊게 파고든 뒤 솔버의 침투 보정이
				// 이를 밀어내면서 에너지가 다시 주입돼 "속도가 줄었다 다시 증가"한다.
				// 한 스텝 변위가 반지름의 일정 비율을 넘지 않도록 substep으로 쪼갠다.
				const r = this.config.ball.radius;
				const dt = this.config.physics.timeStep;
				let maxSpeed = 0;
				for (const ball of activeBalls) {
					const v = ball.rigidbody.linvel();
					maxSpeed = Math.max(maxSpeed, Math.hypot(v.x, v.y, v.z));
				}
				const maxDispPerStep = r * 0.25;
				const substeps = Math.min(
					8,
					Math.max(1, Math.ceil((maxSpeed * dt) / maxDispPerStep)),
				);
				// 총 진행 시간은 dt로 동일하게 유지하면서 침투 깊이만 줄인다.
				this.world.timestep = dt / substeps;

				const collidedHandles = new Set<number>();
				for (let i = 0; i < substeps; i++) {
					this.world.step(this.eventQueue);
					this.eventQueue.drainCollisionEvents((h1, h2, started) => {
						if (!started) return;
						collidedHandles.add(h1);
						collidedHandles.add(h2);
					});
				}

				// 지수 감쇠(damping)만으로는 감속량이 속도에 비례해 저속에서
				// 무한히 기어간다. 속도와 무관한 "일정 감속(구름 저항)"을 더해
				// 유한 시간에 멈추게 하고, 임계 속도 이하면 완전히 정지시킨다.
				const decel = this.config.physics.rollingFriction * 9.81; // m/s^2
				const STOP_SPEED = 0.05; // 이 속도(m/s) 이하면 정지로 간주
				for (const ball of activeBalls) {
					const v = ball.rigidbody.linvel();
					const speed = Math.hypot(v.x, v.z);
					if (speed < 1e-6) continue;
					const next = speed - decel * dt;
					if (next <= STOP_SPEED) {
						ball.rigidbody.setLinvel(new Vector3(0, v.y, 0), true);
						ball.rigidbody.setAngvel(new Vector3(0, 0, 0), true);
					} else {
						const f = next / speed;
						ball.rigidbody.setLinvel(new Vector3(v.x * f, v.y, v.z * f), true);
						// 굴림 각속도도 같은 비율로 줄여 v = ω×R 굴림 조건을 유지
						const w = ball.rigidbody.angvel();
						ball.rigidbody.setAngvel(
							new Vector3(w.x * f, w.y * f, w.z * f),
							true,
						);
					}
				}

				return {
					cueBall: {
						position: this.cueBall.rigidbody.translation(),
						rotation: this.cueBall.rigidbody.rotation(),
						linvel: this.cueBall.rigidbody.linvel(),
						angvel: this.cueBall.rigidbody.angvel(),
						radius: this.config.ball.radius,
						collided: collidedHandles.has(this.cueBall.collider.handle),
					},
					objectBalls: this.objectBalls
						.slice(0, objectBallPositions.length)
						.map((ball) => ({
							position: ball.rigidbody.translation(),
							rotation: ball.rigidbody.rotation(),
							linvel: ball.rigidbody.linvel(),
							angvel: ball.rigidbody.angvel(),
							radius: this.config.ball.radius,
							collided: collidedHandles.has(ball.collider.handle),
						})),
				};
			},
		];
	}
}

export default Simulator;
