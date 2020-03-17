import { Cohort } from "../Cohort";
import { JointDataset } from "../JointDataset";
import React from "react";
import { mergeStyleSets } from "@uifabric/styling";
import { DefaultButton } from "office-ui-fabric-react/lib/Button";
import _ from "lodash";
import { Callout } from "office-ui-fabric-react/lib/Callout";
import { FilterControl } from "./FilterControl";

export interface ICohortControlProps {
    cohorts: Cohort[];
    jointDataset: JointDataset;
    onChange: (newCohort: Cohort, index: number) => void;
    onDelete: (index: number) => void;
}

export interface ICohortControlState {
    cohortIndex?: number;
}

export class CohortControl extends React.PureComponent<ICohortControlProps, ICohortControlState> {
    private static styles = mergeStyleSets({
        wrapper: {
            height: "150px",
            width: "100%",
            display: "flex",
            flexDirection: "row"
        },
        createNew: {
            width: "140px",
            height: "100%",
            border:"1px solid grey"
        },
        existingItem: {
            width: "140px",
            height: "100%",
            border:"1px solid grey"
        }
    });

    constructor(props: ICohortControlProps) {
        super(props);
        this.state = {};
        this.onCancel = this.onCancel.bind(this);
        this.openDialog = this.openDialog.bind(this);
    }
    public render(): React.ReactNode {
        let cohortForEdit: Cohort;
        if (this.state.cohortIndex !== undefined) {
            if (this.state.cohortIndex === this.props.cohorts.length) {
                cohortForEdit = new Cohort("cohort " + this.state.cohortIndex, this.props.jointDataset);
            } else {
                cohortForEdit = _.cloneDeep(this.props.cohorts[this.state.cohortIndex]);
            }
        }
        return (
            <div className={CohortControl.styles.wrapper}>
                {cohortForEdit !== undefined && (
                    <Callout
                        onDismiss={this.onCancel}
                        setInitialFocus={true}
                        hidden={false}
                    >
                        <div>{cohortForEdit.name}</div>
                        <FilterControl 
                            jointDataset={this.props.jointDataset}
                            filterContext={{
                                filters: cohortForEdit.filters,
                                onAdd: (filter) => {cohortForEdit.updateFilter(filter)},
                                onDelete: cohortForEdit.deleteFilter,
                                onUpdate: (filter, index) => {cohortForEdit.updateFilter(filter, index)}
                            }}
                        />
                        <DefaultButton onClick={this.updateCohort.bind(this, cohortForEdit)}>Accept changes</DefaultButton>
                    </Callout>
                )}
                <div className={CohortControl.styles.createNew}>
                    <DefaultButton onClick={this.openDialog.bind(this, undefined)} text={"Create new cohort"}/>
                </div>
                {this.props.cohorts.map((cohort, index) => {
                    return (<div 
                        className={CohortControl.styles.existingItem}
                        onClick={this.openDialog.bind(this, index)}>{cohort.name}</div>)
                })}
            </div>
        );
    }

    private onCancel(): void {
        this.setState({cohortIndex: undefined});
    }

    private updateCohort(newCohort: Cohort): void {
        this.props.onChange(newCohort, this.state.cohortIndex);
        this.setState({cohortIndex: undefined});
    }

    private openDialog(cohortIndex?: number): void {
        if (cohortIndex === undefined) {
            this.setState({cohortIndex: this.props.cohorts.length});
        } else {
            this.setState({cohortIndex});
        }
    }
}