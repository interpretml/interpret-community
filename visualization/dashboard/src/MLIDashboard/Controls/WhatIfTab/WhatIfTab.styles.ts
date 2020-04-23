import { IProcessedStyleSet, getTheme, mergeStyleSets, IStyle } from "@uifabric/styling";

export interface IWhatIfTabStyles {
    page: IStyle;
    blackIcon: IStyle;
    expandedPanel: IStyle;
    parameterList: IStyle;
    featureList: IStyle;
    collapsedPanel: IStyle;
    mainArea: IStyle;
    infoIcon: IStyle;
    helperText: IStyle;
    infoWithText: IStyle;
    chartsArea: IStyle;
    topArea: IStyle;
    legendAndText: IStyle;
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
    featureImportanceLegend: IStyle;
    featureImportanceChartAndLegend: IStyle;
    legendHelpText: IStyle;
    legendLabel: IStyle;
    smallItalic: IStyle;
    legendHlepWrapper: IStyle;
    choiceBoxArea: IStyle;
    choiceGroup: IStyle;
    choiceGroupFlexContainer: IStyle;
    panelIconAndLabel: IStyle;
    secondaryChartPlacolderBox: IStyle;
    secondaryChartPlacolderSpacer: IStyle;
    faintText: IStyle;
    predictedBlock: IStyle;
    upperWhatIfPanel: IStyle;
    saveButton: IStyle;
    customPredictBlock: IStyle;
}

export const whatIfTabStyles: () => IProcessedStyleSet<IWhatIfTabStyles> = () => {
    const legendWidth = "160px";
    const theme = getTheme();
    return mergeStyleSets<IWhatIfTabStyles>({
        page: {
            width: "100%",
            padding: "16px 0 0 14px",
            boxSizing: "border-box"
        },
        blackIcon: {
            color: theme.semanticColors.bodyText
        },
        expandedPanel: {
            marginTop: "10px",
            width: "250px",
            boxShadow: "0px 4.8px 14.4px rgba(0, 0, 0, 0.18), 0px 25.6px 57.6px rgba(0, 0, 0, 0.22)",
            display: "flex",
            flexDirection: "column"
        },
        parameterList: {
            margin: "8px 18px 30px 22px",
            display: "flex",
            flexGrow: 1,
            backgroundColor: theme.palette.neutralLighter,
            padding: "6px",
            flexDirection: "column"
        },
        featureList: {
            display: "flex",
            paddingLeft: "10px",
            flexGrow: 1,
            flexDirection: "column",
            maxHeight: "400px",
            overflowY: "auto"
        },
        collapsedPanel: {
            width: "40px",
            boxShadow: "0px 4.8px 14.4px rgba(0, 0, 0, 0.18), 0px 25.6px 57.6px rgba(0, 0, 0, 0.22)"
        },
        mainArea: {
            flex: 1,
            display: "flex",
            flexDirection: "row-reverse",
            alignItems: "stretch",
            minHeight: "800px"
        },
        chartsArea: {
            flex: 1
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
        topArea: {
            width: "100%",
            height: "400px",
            display: "flex",
            flexDirection: "row"
        },
        legendAndText: {
            height: "100%",
            width: legendWidth,
            boxSizing: "border-box",
            paddingLeft: "10px",
            paddingRight: "10px"

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
            width: "100%"
        },
        featureImportanceChartAndLegend: {
            height: "300px",
            width: "100%",
            display: "flex",
            flexDirection: "row"
        },
        featureImportanceLegend: {
            height: "100%",
            width: legendWidth
        },
        sliderLabel: {
            fontWeight: "600",
            paddingRight: "10px"
        },
        startingK: {
            flex: 1,
            paddingRight: legendWidth
        },
        featureImportanceControls: {
            display: "flex",
            flexDirection: "row",
            padding: "18px 30px 4px 67px",

        },
        legendHlepWrapper: {
            width: "120px"
        },
        legendHelpText: {
            fontWeight: "300",
            lineHeight: "14px",
            width: "120px"
        },
        legendLabel: {
            fontWeight: "600",
            paddingBottom: "5px",
            paddingTop: "10px"
        },
        smallItalic: {
            fontStyle: "italic",
            padding: "0 0 5px 5px",
            color: theme.semanticColors.disabledBodyText
        },
        choiceBoxArea: {
            paddingLeft: "67px",
            display: "flex",
            flexDirection: "row",
            alignItems: "baseline"
        },
        choiceGroup: {
            fontSize: "14px",
            paddingLeft: "30px",
            width: "230px",

        },
        choiceGroupFlexContainer: {
            display: "inline-flex",
            width: "300px",
            justifyContent: "space-between"
        },
        panelIconAndLabel: {
            paddingTop: "10px",
            alignItems: "center",
            display: "flex"
        },
        secondaryChartPlacolderBox: {
            height: "300px",
            width: "100%"
        },
        secondaryChartPlacolderSpacer: {
            margin: "25px auto 0 auto",
            padding: "23px",
            width:"fit-content",
            boxShadow: "0px 0px 6px rgba(0, 0, 0, 0.2)"
        },
        faintText: {
            fontWeight: "350" as any,
        },
        predictedBlock: {
            paddingLeft: "6px",
            paddingTop: "5px"
        },
        upperWhatIfPanel: {
            paddingLeft: "32px",
            paddingRight: "32px"
        },
        customPredictBlock: {
            paddingLeft: "24px",
            paddingBottom: "10px"
        },
        saveButton: {
            margin: "0 0 10px 24px"
        }
    });
}