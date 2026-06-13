import { createTheme } from "@vanilla-extract/css";

export const [theme, vars] = createTheme({
	color: {
		background: "black",
		onBackground: "white",
		surface: "#222",
		onSurface: "white",
		text: "white",
        textAccent: "#00e5ff",
	},
	font: {
		body: `"Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, system-ui, Roboto, "Helvetica Neue", "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif`,
	},
    fontSize: {
        body: "16px",
        title: "24px",
        subtitle: "12px",
    },
	browser: {
		colorScheme: "dark",
	},
});
