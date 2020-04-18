import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";

export interface IGlobalTabStyles {
    page: IStyle;
    infoIcon: IStyle;
    helperText: IStyle;
    infoWithText: IStyle;
    globalChartControls: IStyle;
    sliderLabel: IStyle;
    topK: IStyle;
    startingK: IStyle;
    chartTypeDropdown: IStyle;
    globalChartWithLegend: IStyle;
    legendAndSort: IStyle;
    cohortLegend: IStyle;
    legendHelpText: IStyle;
    secondaryChartAndLegend: IStyle;
}

export const globalTabStyles: () => IProcessedStyleSet<IGlobalTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IGlobalTabStyles>({
        page: {
            width: "100%",
            height: "100%",
            padding: "16px 40px 0 14px",
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
            paddingRight: "120px",
            paddingLeft: "15px"
        },
        globalChartControls: {
            display: "flex",
            flexDirection: "row",
            padding: "18px 300px 4px 67px",
            
        },
        sliderLabel: {
            fontWeight: "600",
            paddingRight: "10px"
        },
        topK: {
            maxWidth: "200px"
        },
        startingK: {
            flex: 1
        },
        chartTypeDropdown: {
            margin: "0 5px 0 0"
        },
        globalChartWithLegend: {
            height: "400px",
            width: "100%",
            display: "flex",
            flexDirection: "row"
        },
        secondaryChartAndLegend: {
            height: "300px",
            width: "100%",
            display: "flex",
            flexDirection: "row"
        },
        legendAndSort: {
            width: "200px",
            height: "100%"
        },
        cohortLegend: {
            fontWeight: "600",
            paddingBottom: "10px"
        },
        legendHelpText: {
            fontWeight: "300"
        }
    });
};