import { todo } from "@/common";
import hyperparams from "@/config/hyperparams";

export class TrajectoryPainter {
	private readonly prepassCanvasHandle: CanvasHandle<"2d">;

	public constructor(width: number, height: number) {
		const canvas = new OffscreenCanvas(width, height);
		const context = canvas.getContext("2d") ?? todo("2d context not supported");

		this.prepassCanvasHandle = {
			canvas,
			draw: (pass) => pass(context, width, height),
		};
	}

	private drawTrajetory(
		trajectory: BallTrajectory,
		color: { r: number; g: number; b: number },
		scale: number,
		showOutline: boolean = false,
	) {
		this.prepassCanvasHandle.draw((context, width, height) => {
			context.clearRect(0, 0, width, height);

			if (trajectory.snapshots.length === 0) {
				return;
			}

			const initialTrajactory = trajectory.snapshots[0];

			if (showOutline) {
				context.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 1)`;
				context.lineWidth = width * 0.003;
				context.beginPath();
				context.arc(
					initialTrajactory.position.x * scale,
					initialTrajactory.position.z * scale,
					hyperparams.ball.radius * scale,
					0,
					2 * Math.PI,
				);
				context.stroke();
			}

			const maxWidth = hyperparams.ball.radius * scale;
			const minWidth = maxWidth * 0.15;

			context.lineJoin = "round";
			context.lineCap = "round";
			context.strokeStyle = "rgba(255, 255, 255, 1)";

			let prevX = initialTrajactory.position.x * scale;
			let prevY = initialTrajactory.position.z * scale;
			for (let i = 1; i < trajectory.snapshots.length; i++) {
				const snapshot = trajectory.snapshots[i];
				const x = snapshot.position.x * scale;
				const y = snapshot.position.z * scale;

				const t =
					trajectory.snapshots.length > 2
						? (i - 1) / (trajectory.snapshots.length - 2)
						: 0;

				context.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${1 - t})`;
				context.lineWidth = maxWidth + (minWidth - maxWidth) * (t);

				context.beginPath();
				context.moveTo(prevX, prevY);
				context.lineTo(x, y);
				context.stroke();

				prevX = x;
				prevY = y;
			}
		});
	}

	public drawTrajectories(
		canvasHandle: CanvasHandle<"2d">,
		snapshots: TableSnapshot[],
		scale: number = 1000,
		showOutline = false,
	) {
		if (snapshots.length === 0) {
			return;
		}

		const initialSnapshot = snapshots[0];

		const trajectories = {
			cueBall: {
				snapshots: snapshots.map(
					(snapshot) => snapshot.cueBall,
				) as BallSnapshot[],
			},
			objectBalls: initialSnapshot.objectBalls.map(
				(_, index) =>
					({
						snapshots: snapshots.map((snapshot) => snapshot.objectBalls[index]),
					}) as BallTrajectory,
			),
		};

		canvasHandle.draw((context, width, height) => {
			context.clearRect(0, 0, width, height);

			if (snapshots.length === 0) {
				return;
			}

			this.drawTrajetory(
				trajectories.cueBall,
				{ r: 255, g: 255, b: 255 },
				scale,
				showOutline,
			);

			context.shadowBlur = width * 0.05;
			context.shadowColor = "rgba(255, 255, 255, 0.8)";
			context.drawImage(this.prepassCanvasHandle.canvas, 0, 0);

			for (let i = 0; i < trajectories.objectBalls.length; i++) {
				this.drawTrajetory(
					trajectories.objectBalls[i],
					{ r: 255, g: 125, b: 125 },
					scale,
					showOutline,
				);

				context.shadowColor = "rgba(255, 125, 125, 0.8)";
				context.drawImage(this.prepassCanvasHandle.canvas, 0, 0);
			}
		});
	}
}
