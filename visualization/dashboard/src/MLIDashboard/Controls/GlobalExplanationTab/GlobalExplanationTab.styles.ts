import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";

export interface IGlobalTabStyles {
  page: IStyle;
  globalChartControls: IStyle;
  topK: IStyle;
  startingK: IStyle;
  chartTypeDropdown: IStyle;
  globalChartWithLegend: IStyle;
  legendAndSort: IStyle;
}

export const globalTabStyles: () => IProcessedStyleSet<IGlobalTabStyles> = () => {
    const theme = getTheme();
  return mergeStyleSets<IGlobalTabStyles>({
    page: {
        height: "100%",
        display: "flex",
        flexDirection: "column",
        width: "100%",
        backgroundColor: theme.semanticColors.bodyStandoutBackground
    },
    globalChartControls: {
        display: "flex",
        flexDirection: "row"
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
    legendAndSort: {
        width: "200px"
    }
  });
};