import {
	type ChangeEventHandler,
	type PointerEventHandler,
	useCallback,
	useState,
} from "react";
import { styles } from "./index.css";

type HitControlPanelProps = {
	/**
	 *
	 * @param point [-1, 1] 범위
	 * @returns
	 */
	onHitPointChange: (point: Vector2<"unit">) => void;
	onHitPowerChange: (power: number) => void;
	/**
	 *
	 * @param angle radian
	 * @returns
	 */
	onHitAngleChange?: (angle: number) => void;
};

/**
 * hit point와 hit power를 조절할 수 있는 컨트롤 패널 컴포넌트
 * @param props
 * @returns
 */
function HitControlPanel(props: HitControlPanelProps) {
	const [hitPoint, setHitPoint] = useState<Vector2<"unit">>({ x: 0, y: 0 });
	const [hitPower, setHitPower] = useState(1.5);
	const [hitAngle, setHitAngle] = useState(0);

	const onHitPointChange = useCallback<PointerEventHandler<HTMLDivElement>>(
		(event) => {
			if (event.buttons === 0) {
				return;
			}

			const rect = event.currentTarget.getBoundingClientRect();
			const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
			const y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);

			const angle = Math.atan2(y, x);
			const maxX = Math.abs(Math.cos(angle));
			const maxY = Math.abs(Math.sin(angle));

			const clampedX = Math.max(-maxX, Math.min(maxX, x));
			const clampedY = Math.max(-maxY, Math.min(maxY, y));

			setHitPoint({ x: clampedX, y: clampedY });
			props.onHitPointChange({ x: clampedX, y: clampedY });
		},
		[props],
	);
	const onHitPowerChange = useCallback<
		ChangeEventHandler<HTMLInputElement, HTMLInputElement>
	>(
		(event) => {
			const power = parseFloat(event.target.value);
			setHitPower(power);
			props.onHitPowerChange(power);
		},
		[props],
	);
	const onHitAngleChange = useCallback<PointerEventHandler<HTMLDivElement>>(
		(event) => {
			if (event.buttons === 0) {
				return;
			}

			const rect = event.currentTarget.getBoundingClientRect();
			const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
			const y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);

			const angle = Math.atan2(y, x) + Math.PI;
			setHitAngle(angle);
			props.onHitAngleChange?.(angle);
		},
		[props],
	);
	const resetHitPoint = useCallback(() => {
		setHitPoint({ x: 0, y: 0 });
		props.onHitPointChange({ x: 0, y: 0 });
	}, [props]);

	return (
		<div className={styles.root}>
			<label
				style={{
					width: "100%",
					display: "flex",
					flexDirection: "column",
					gap: "8px",
				}}
			>
				<span>{`Power ${hitPower.toFixed(2)}`}</span>
				<input
					style={{
						width: "100%",
					}}
					type="range"
					min="0"
					max="4"
					step="0.001"
					value={hitPower}
					onChange={onHitPowerChange}
				/>
			</label>

			<div
				style={{
					width: "100%",
					display: "flex",
					flexDirection: "column",
					alignItems: "center",
					gap: "8px",
				}}
			>
				<span
					style={{
						width: "100%",
					}}
				>{`Spin side ${hitPoint.x.toFixed(2)} / top ${hitPoint.y.toFixed(2)}`}</span>

				<div
					style={{
						width: "auto",
						display: "flex",
						flexDirection: "row",
						alignItems: "center",
						gap: "16px",
					}}
				>
					<div
						style={{
							width: "96px",
							height: "auto",
							aspectRatio: "1 / 1",
							borderRadius: "50%",
							background:
								"radial-gradient(circle at 38% 32%, rgba(255, 255, 255, 0.95), rgba(230, 238, 242, 0.9) 35%, rgba(165, 181, 190, 0.92) 100%)",
							boxShadow:
								"inset -10px -12px 18px rgba(0, 0, 0, 0.2), inset 8px 8px 14px rgba(255, 255, 255, 0.65), 0 0 0 2px rgba(255, 255, 255, 0.18)",
							position: "relative",
							touchAction: "none",
						}}
						onPointerDown={onHitPointChange}
						onPointerMove={onHitPointChange}
					>
						<div
							style={{
								position: "absolute",
								width: "64%",
								height: "60%",
								top: "50%",
								left: "50%",
								transform: "translate(-50%, -50%)",
								border: "1px dashed rgba(0, 0, 0, 0.28)",
								borderRadius: "50%",
							}}
						/>
						<div
							style={{
								position: "absolute",
								width: "72%",
								height: "1px",
								top: "50%",
								left: "50%",
								transform: "translate(-50%, -50%)",
								border: "1px dashed rgba(0, 0, 0, 0.28)",
							}}
						/>
						<div
							style={{
								position: "absolute",
								width: "1px",
								height: "72%",
								top: "50%",
								left: "50%",
								transform: "translate(-50%, -50%)",
								border: "1px dashed rgba(0, 0, 0, 0.28)",
							}}
						/>
						<div
							style={{
								position: "absolute",
								width: "10%",
								height: "10%",
								borderRadius: "50%",
								backgroundColor: "rgba(255, 0, 0, 0.8)",
								border: "2px solid rgba(255, 255, 255, 1)",
								top: `${-hitPoint.y * 50 + 50}%`,
								left: `${hitPoint.x * 50 + 50}%`,
								transform: "translate(-50%, -50%)",
							}}
						/>
					</div>

					<button
						style={{
							width: "auto",
							height: "auto",
							padding: "8px 16px",
							borderRadius: "8px",
							fontWeight: "800",
							cursor: "pointer",
						}}
						type="button"
						onClick={resetHitPoint}
					>
						Center
					</button>
				</div>
			</div>

			{props.onHitAngleChange && (
				<div
					style={{
						width: "100%",
						display: "flex",
						flexDirection: "column",
						alignItems: "center",
						gap: "8px",
					}}
				>
					<span
						style={{
							width: "100%",
						}}
					>{`Angle: ${((hitAngle * 180) / Math.PI).toFixed(2)}`}</span>

					<div
						style={{
							width: "60%",
							height: "auto",
							aspectRatio: "1 / 1",
							borderRadius: "50%",
							position: "relative",
							touchAction: "none",
						}}
						onPointerDown={onHitAngleChange}
						onPointerMove={onHitAngleChange}
					>
						<div
							style={{
								position: "absolute",
								width: "80%",
								height: "80%",
								top: "50%",
								left: "50%",
								transform: "translate(-50%, -50%)",
								border: "1px dashed rgba(255, 255, 255, 0.56)",
								borderRadius: "50%",
							}}
						/>

						<div
							style={{
								position: "absolute",
								width: "100%",
								height: "100%",
								rotate: `${-hitAngle}rad`,
							}}
						>
							<div
								style={{
									position: "absolute",
									width: "72%",
									height: "1px",
									top: "50%",
									left: "50%",
									transform: "translate(-50%, -50%)",
									border: "1px solid rgba(255, 255, 255, 0.56)",
								}}
							/>
							<div
								style={{
									position: "absolute",
									width: "10%",
									height: "10%",
									top: "50%",
									left: "80%",
									transform: "translate(-50%, -50%) rotate(45deg)",
									borderTop: "2px solid rgba(255, 255, 255, 0.56)",
									borderRight: "2px solid rgba(255, 255, 255, 0.56)",
								}}
							/>
						</div>
					</div>
				</div>
			)}
		</div>
	);
}

export default HitControlPanel;
