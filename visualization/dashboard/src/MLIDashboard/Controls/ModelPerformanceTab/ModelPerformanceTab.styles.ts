import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";

export interface IModelPerformanceTabStyles {
    page: IStyle;
    infoIcon: IStyle;
    helperText: IStyle;
    infoWithText: IStyle;
    scrollableWrapper: IStyle;
    scrollContent: IStyle;
    chartWithAxes: IStyle;
    chartWithVertical: IStyle;
    verticalAxis: IStyle;
    rotatedVerticalBox: IStyle;
    chart: IStyle;
    rightPanel: IStyle;
    statsBox: IStyle;
    horizontalAxisWithPadding: IStyle;
    paddingDiv: IStyle;
    horizontalAxis: IStyle;
    cohortPickerWrapper: IStyle;
    cohortPickerLabel: IStyle;
    boldText: IStyle;
}

export const modelPerformanceTabStyles: () => IProcessedStyleSet<IModelPerformanceTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IModelPerformanceTabStyles>({
        page: {
            width: "100%",
            height: "100%",
            padding: "16px 0 0 14px",
            boxSizing: "border-box",
            display: "flex",
            flexDirection: "column"
        },
        infoWithText: {
            display: "flex",
            flexDirection: "row",
            width: "100%",
            boxSizing: "border-box",
            paddingLeft: "25px"
        },
        infoIcon: {
            width: "23px",
            height: "23px",
            fontSize: "23px"
        },
        helperText: {
            paddingRight: "160px",
            paddingLeft: "15px"
        },
        chartWithAxes: {
            display: "flex",
            flexGrow: "1",
            boxSizing: "border-box",
            paddingTop: "30px",
            flexDirection: "column"
        },
        chartWithVertical: {
            display: "flex",
            flexGrow: "1",
            flexDirection: "row"
        },
        scrollableWrapper: {
            flexGrow: "1",
            overflowY: "auto"
        },
        scrollContent: {
            width: "100%",
            display: "flex",
            flexDirection:"row",
            alignItems: "stretch",
            height: "500px"
        },
        chart: {
            flexGrow: "1"
        },
        rightPanel: {
            width: "195px",
            height: "100%",
            backgroundColor: theme.semanticColors.bodyBackgroundHovered,
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-evenly",
            padding: "40px 15px 30px 30px"
        },
        statsBox: {
            padding: "23px 30px 30px 30px",
            boxShadow: "0px 4px 4px rgba(0, 0, 0, 0.25)",
            backgroundColor: theme.semanticColors.bodyBackground
        },
        verticalAxis: {
            position: "relative",
            top: "0px",
            height: "auto",
            width: "64px"
        },
        rotatedVerticalBox: {
            transform: "translateX(-50%) translateY(-50%) rotate(270deg)",
            marginLeft: "28px",
            position: "absolute",
            top: "50%",
            textAlign: "center",
            width: "max-content"
        },
        horizontalAxisWithPadding: {
            display: "flex",
            paddingBottom: "30px",
            flexDirection: "row"
        },
        paddingDiv: {
            width: "50px"
        },
        horizontalAxis: {
            flex: 1,
            textAlign:"center"
        },
        cohortPickerWrapper: {
            paddingLeft: "63px",
            paddingTop: "13px",
            height: "32px",
            display: "flex",
            flexDirection: "row",
            alignItems: "center"
        },
        cohortPickerLabel: {
            fontWeight: "600",
            paddingRight: "8px"
        },
        boldText: {
            fontWeight: "600",
            paddingBottom: "5px"
        }
    });
};