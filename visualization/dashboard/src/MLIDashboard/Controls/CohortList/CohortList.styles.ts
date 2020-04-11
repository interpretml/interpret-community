import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";

export interface ICohortListStyles {
  banner: IStyle;
  summaryLabel: IStyle;
  mediumText: IStyle;
  summaryBox: IStyle;
  summaryItemText: IStyle;
  chartTypeDropdown: IStyle;
  globalChartWithLegend: IStyle;
  legendAndSort: IStyle;
}

export const cohortListStyles: () => IProcessedStyleSet<ICohortListStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ICohortListStyles>({
        banner: {
            height: "105px",
            paddingTop: "5px",
            paddingLeft: "34px",
            display: "flex",
            flexDirection: "row",
            width: "100%",
            color: theme.palette.white,
            backgroundColor: theme.palette.neutralPrimary
        },
        summaryLabel: {
            fontVariant: "small-caps",
            display: "flex",
            flexDirection: "row"
        },
        mediumText: {
            maxWidth: "200px"
        },
        summaryBox: {
            width: "125px",
        },
        summaryItemText: {

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