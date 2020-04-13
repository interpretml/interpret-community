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
    horizontalAxisWithPadding: IStyle;
    paddingDiv: IStyle;
    horizontalAxis: IStyle;
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
            backgroundColor: theme.palette.neutralLighter
        },
        verticalAxis: {
            position: "relative",
            top: "0px",
            height: "auto",
            width: "50px"
        },
        rotatedVerticalBox: {
            transform: "translateX(-50%) translateY(-50%) rotate(270deg)",
            marginLeft: "15px",
            position: "absolute",
            top: "50%",
            textAlign: "center",
            width: "max-content"
        },
        horizontalAxisWithPadding: {
            display: "flex",
            flexDirection: "row"
        },
        paddingDiv: {
            width: "50px"
        },
        horizontalAxis: {
            flex: 1,
            textAlign:"center"
        }
    });
};