import { Cohort } from "../Cohort";
import { JointDataset } from "../JointDataset";
import React from "react";
import { mergeStyleSets } from "@uifabric/styling";
import { DefaultButton, PrimaryButton } from "office-ui-fabric-react/lib/Button";
import _ from "lodash";
import { Callout, ICalloutContentStyles } from "office-ui-fabric-react/lib/Callout";
import { FilterControl } from "./FilterControl";
import { TextField } from "office-ui-fabric-react/lib/TextField";
import { getId } from "office-ui-fabric-react/lib/Utilities";
import { IStyle } from "office-ui-fabric-react/lib/Styling";

//const cohortNameId = getId('cohortNameId')

let calloutOverrideProps: IStyle = {
    position: 'absolute',
    overflowY: 'visible',
    width:'560px',
    height:'455px',
    left:'212px',
    top:'219px',
    background: '#FFFFFF',
    boxShadow: '0px 0.6px 1.8px rgba(0, 0, 0, 0.108), 0px 3.2px 7.2px rgba(0, 0, 0, 0.132)',
    borderRadius: '2px'
}

let calloutMainProps: ICalloutContentStyles = {
    container:{},
    root: {},
    beak: {},
    beakCurtain: {},
    calloutMain:calloutOverrideProps
}

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
        },
        cohortName: {
            position:"absolute",
            width:"35%",
            height:"56px"
        },
        saveCohort: {
            display: 'flex',
            flexDirection: 'row',
            padding: '6px 16px',
            position: 'absolute',
            width: '62px',
            height: '32px',
            left: '471px',
            top: '396px',
            background: '#F3F2F1',
            borderRadius: '2px'
        },
        Addfilter:{
            position:'absolute',
            width:'100%',
            top: '65px',
            left:'50px'
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
                cohortForEdit = new Cohort("janhavi" + this.state.cohortIndex, this.props.jointDataset);
            } else {
                cohortForEdit = _.cloneDeep(this.props.cohorts[this.state.cohortIndex]);
                console.log("cohort name here", this.props.cohorts[this.state.cohortIndex])
            }
        }
        return (
             <div className={CohortControl.styles.wrapper}>
                    {cohortForEdit !== undefined && (
                        <Callout
                            onDismiss={this.onCancel}
                            setInitialFocus={true}
                            hidden={false}
                            styles={calloutMainProps}
                        >

                        <div className={CohortControl.styles.cohortName}>
                        <label id={cohortForEdit.getCohortID.toString()}>Dataset cohort name</label>
                        <TextField id={cohortForEdit.getCohortID.toString()} 
                        placeholder="Enter dataset cohort name"></TextField>
                        </div>
                    
                    <div className={CohortControl.styles.Addfilter}>
                        <FilterControl
                                jointDataset={this.props.jointDataset}
                                filterContext={{
                                    filters: cohortForEdit.filters,
                                    onAdd: (filter) => {cohortForEdit.updateFilter(filter)},
                                    onDelete: cohortForEdit.deleteFilter,
                                    onUpdate: (filter, index) => {cohortForEdit.updateFilter(filter, index)}
                                }}
                            />
                    </div>
                                                
                        <PrimaryButton className={CohortControl.styles.saveCohort} onClick={this.updateCohort.bind(this, cohortForEdit)}>Save</PrimaryButton> 
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