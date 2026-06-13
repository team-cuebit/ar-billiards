import { globalStyle } from "@vanilla-extract/css";
import { vars } from "./theme.css";

globalStyle("html, body", {
    interpolateSize: "allow-keywords",
	backgroundColor: vars.color.background,
	color: vars.color.text,
	colorScheme: vars.browser.colorScheme,
	fontFamily: vars.font.body,
	fontSize: vars.fontSize.body,
});

globalStyle("button", {
	width: "auto",
	height: "auto",
	padding: "8px 0px",
	fontWeight: "bold",
	borderRadius: "8px",
	cursor: "pointer",
	textTransform: "uppercase",
	border: "none",
	overflow: "hidden",
	textWrap: "nowrap",
});
