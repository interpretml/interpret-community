import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";

export interface ICohortListStyles {
  banner: IStyle;
  summaryLabel: IStyle;
  mediumText: IStyle;
  summaryBox: IStyle;
  summaryItemText: IStyle;
  cohortList: IStyle;
  cohortBox: IStyle;
  cohortLabelWrapper: IStyle;
  cohortLabel: IStyle;
  overflowButton: IStyle;
}

export const cohortListStyles: () => IProcessedStyleSet<ICohortListStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ICohortListStyles>({
        banner: {
            height: "105px",
            paddingTop: "10px",
            paddingLeft: "34px",
            display: "flex",
            flexDirection: "row",
            width: "100%",
            color: theme.palette.white,
            backgroundColor: theme.palette.neutralPrimary
        },
        summaryLabel: {
            fontVariant: "small-caps",
            marginBottom: "2px"
        },
        mediumText: {
            maxWidth: "200px"
        },
        summaryBox: {
            width: "141px",
        },
        summaryItemText: {

        },
        cohortList: {
        },
        cohortBox: {
            width: "120px",
            display: "inline-block"
        },
        cohortLabelWrapper: {
            maxWidth: "100%",
            display: "flex",
            flexDirection: "row"
        },
        cohortLabel: {
            flexGrow: 1
        },
        overflowButton: {
            backgroundColor: theme.palette.neutralPrimary,
            border: "none"
        }
    });
};