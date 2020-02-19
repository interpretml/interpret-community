import React from "react";
import { JointDataset } from "../JointDataset";
import { IExplanationModelMetadata } from "../IExplanationContext";
import { IFilterContext } from "../Interfaces/IFilter";

export interface IGlobalBarSettings {
    topK: number;
    sortOption: string;
    includeOverallGlobal: boolean;
}

export interface IGlobalExplanationTabProps {
    globalBarSettings: IGlobalBarSettings;
    // selectionContext: SelectionContext;
    theme?: string;
    // messages?: HelpMessageDict;
    jointDataset: JointDataset;
    metadata: IExplanationModelMetadata;
    filterContext: IFilterContext;
    onChange: (props: IGlobalBarSettings) => void;
}

export class GlobalExplanationTab extends React.PureComponent<IGlobalExplanationTabProps> {
    constructor(props: IGlobalExplanationTabProps) {
        super(props);
    }
    public render(): React.ReactNode {
        return(<></>);
    }
}