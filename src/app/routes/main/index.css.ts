import { keyframes, style } from "@vanilla-extract/css";

export const styles = {
	root: style({
		width: "100vw",
		height: "100vh",
		display: "flex",
		overflow: "hidden",
		position: "relative",
		touchAction: "none",
	}),
	button: style({
		minWidth: 0,
	}),
};

export const animation = {
	blink: keyframes({
		"0%": {
			opacity: 1,
		},
		"50%": {
			opacity: 0.3,
		},
		"100%": {
			opacity: 1,
		},
	}),
};
