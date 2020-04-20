import { IProcessedStyleSet, getTheme, mergeStyleSets, IStyle } from "@uifabric/styling";

export interface IWhatIfTabStyles {
    page: IStyle;
    expandedPanel: IStyle;
    parameterList: IStyle;
    featureList: IStyle;
    customPointsList: IStyle;
    collapsedPanel: IStyle;
    mainArea: IStyle;
    infoIcon: IStyle;
    helperText: IStyle;
    infoWithText: IStyle;
    cohortPickerWrapper: IStyle;
    cohortPickerLabel: IStyle;
    boldText: IStyle;
    chartWithAxes: IStyle;
    chartWithVertical: IStyle;
    verticalAxis: IStyle;
    rotatedVerticalBox: IStyle;
    horizontalAxisWithPadding: IStyle;
    paddingDiv: IStyle;
    horizontalAxis: IStyle;
    featureImportanceArea: IStyle;
    sliderLabel: IStyle;
    startingK: IStyle;
    featureImportanceControls: IStyle;
}

export const whatIfTabStyles: () => IProcessedStyleSet<IWhatIfTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IWhatIfTabStyles>({
        page: {
            width: "100%",
            height: "100%",
            padding: "16px 40px 0 14px",
            boxSizing: "border-box",
            display: "flex",
            flexDirection: "row"
        },
        expandedPanel: {
            width: "250px",
            height: "100%",
            borderRight: "1px solid black",
            display: "flex",
            flexDirection: "column"
        },
        parameterList: {
            display: "flex",
            flexGrow: 1,
            flexDirection: "column"
        },
        featureList: {
            display: "flex",
            flexGrow: 1,
            flexDirection: "column",
            maxHeight: "400px",
            overflowY: "auto"
        },
        customPointsList: {
            borderTop: "2px solid black",
            height: "250px",
        },
        collapsedPanel: {
            width: "40px",
            height: "100%",
            borderRight: "1px solid black"
        },
        mainArea: {
            flex: 1,
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
            paddingRight: "120px",
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
            textAlign: "center"
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
        },
        featureImportanceArea: {
            height: "400px",
            width: "100%",

        },
        sliderLabel: {
            fontWeight: "600",
            paddingRight: "10px"
        },
        startingK: {
            flex: 1
        },
        featureImportanceControls: {
            display: "flex",
            flexDirection: "row",
            padding: "18px 300px 4px 67px",

        },
    });
}