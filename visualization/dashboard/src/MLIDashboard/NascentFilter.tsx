import React from "react";
import { IJointMeta, IFilter, FilterMethods } from "./JointDataset";
import { Button } from "office-ui-fabric-react/lib/Button";

export interface INascentFilterProps {
    metaDict: {[key: string]: IJointMeta};
    addFilter: (filter: IFilter) => void;
}

export interface INascentFitlerState {
    selectedColumn: string;
    selectedOperation: FilterMethods;
    value: number | number[];
}

export default class NascentFilter extends React.PureComponent<INascentFilterProps, IFilter> {
    public render(): React.ReactNode {
        return (
            <>
                <Button
                    onClick={this.onClick}
                />
            </>
        );
    }

    // private isValid(): boolean {
    //     return this.state.selectedColumn !== undefined &&
    //         this.state.selectedOperation !== undefined &&
    //         this.state.value !== undefined;
    // }

    private readonly onClick = (): void => {
        this.props.addFilter(this.state);
    }
}