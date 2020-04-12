import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";

export interface IExplanationDashboardStyles {
  pivotLabelWrapper: IStyle;
}

export const explanationDashboardStyles: () => IProcessedStyleSet<IExplanationDashboardStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IExplanationDashboardStyles>({
        pivotLabelWrapper: {
            justifyContent: "space-between",
            display: "flex",
            flexDirection: "row",
            padding: "0 30px"
        }
    });
};