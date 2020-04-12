import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";

export interface IModelPerformanceTabStyles {
    page: IStyle;
    infoIcon: IStyle;
    helperText: IStyle;
    infoWithText: IStyle;
}

export const modelPerformanceTabStyles: () => IProcessedStyleSet<IModelPerformanceTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IModelPerformanceTabStyles>({
        page: {
            width: "100%",
            height: "100%",
            padding: "16px 0 0 14px",
            boxSizing: "border-box"
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
        }
    });
};