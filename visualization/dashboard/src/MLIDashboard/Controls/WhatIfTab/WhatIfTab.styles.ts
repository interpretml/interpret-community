import { IProcessedStyleSet, getTheme, mergeStyleSets, IStyle } from "@uifabric/styling";

export interface IWhatIfTabStyles {
    page: IStyle;
    expandedPanel: IStyle;
    parameterList: IStyle;
    featureList: IStyle;
    collapsedPanel: IStyle;
    mainArea: IStyle;
    infoIcon: IStyle;
    helperText: IStyle;
    infoWithText: IStyle;
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
    secondaryChartChoiceLabel: IStyle;
    choiceBoxArea: IStyle;
    choiceGroup: IStyle;
    choiceGroupFlexContainer: IStyle;
}

export const whatIfTabStyles: () => IProcessedStyleSet<IWhatIfTabStyles> = () => {
    const legendWidth = "150px";
    const theme = getTheme();
    return mergeStyleSets<IWhatIfTabStyles>({
        page: {
            width: "100%",
            height: "100%",
            padding: "16px 40px 0 14px",
            boxSizing: "border-box",
            display: "flex",
            flexDirection: "row-reverse"
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
            paddingLeft: "10px"
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
            height: "350px",
            width: "100%"
        },
        featureImportanceChartAndLegend: {
            flex: "1",
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
            flex: 1
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
            display: "flex",
            flexDirection: "row",
            alignItems: "baseline"
        },
        secondaryChartChoiceLabel: {
            padding: "20px"
        },
        choiceGroup: {
            width: "400px"
        },
        choiceGroupFlexContainer: {
            display: "inline-flex",
            width: "400px",
            justifyContent: "space-between"
        }
    });
}