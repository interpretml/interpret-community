import React from "react";
import { IJointMeta, JointDataset } from "../JointDataset";
import { DefaultButton } from "office-ui-fabric-react/lib/Button";
import { SpinButton } from 'office-ui-fabric-react/lib/SpinButton';
import { FilterMethods, IFilter } from "../Interfaces/IFilter";
import { IComboBoxOption, IComboBox, ComboBox } from "office-ui-fabric-react/lib/ComboBox";
import { localization } from "../../Localization/localization";
import { RangeTypes } from "mlchartlib";
import { Target} from "office-ui-fabric-react/lib/Callout";
import _ from "lodash";
import { mergeStyleSets, FontSizes } from "@uifabric/styling";
import { DetailsList, Selection, SelectionMode, IColumn, CheckboxVisibility } from "office-ui-fabric-react/lib/DetailsList";
import { Checkbox } from "office-ui-fabric-react/lib/Checkbox";
import { Text, FontWeights, getTheme } from "office-ui-fabric-react";
import { Position } from "office-ui-fabric-react/lib/utilities/positioning";

export interface IFilterEditorProps {
    jointDataset: JointDataset;
    initialFilter: IFilter;
    target?: Target;
    onAccept: (filter: IFilter) => void;
    onCancel: () => void;
}

const styles = mergeStyleSets({
    wrapper: {
        height: "295px",
        display: "flex",
        marginTop:"88px"
    },
    leftHalf: {
        height: "295px",
    },
    rightHalf: {
        display: "flex",
        width: "255px",
        height:"295px,",
        flexDirection: "column",
        background: "#F4F4F4",
        marginRight:"27px",
        marginLeft:"25px",
        marginTop:"0px",
        borderRadius: "5px"
    },
    detailedList: {
        marginTop:"28px",
        marginLeft:"40px",
        height:"160px",
        width:"213px"
    },
    selectfilterNav:{
        marginTop:"28px",
        marginLeft:"40px",
        height:"160px",
        width:"213px"
    },
    filterHeader:{
            fontWeight: FontWeights.semibold,
            fontSize: FontSizes.medium,
            color: "#000000"
        },
    addFilterButton: {
        width:"98px",
        height:"32px",
        marginLeft:"32px",
        marginTop:"53px",
        background: "#FFFFFF",
        border: "1px solid #8A8886",
        boxSizing: "border-box",
        borderRadius: "2px"
    },
    featureTextDiv:{
        marginTop:"1px",
    },
    featureComboBox:{
        width:"180px",
        height:"56px",
        marginTop:"21px",
        marginLeft:"30px",
        marginRight:"45px",
        marginBottom:"1px"
    }, 
    operationComboBox:{
        width:"180px",
        height:"56px",
        marginTop:"25px",
        marginLeft:"30px",
        marginRight:"45px",
        marginBottom:"10px"
    },
    valueSpinButton:{
        width:"180px",
        height:"36px",
        marginLeft:"30px",
        marginRight:"45px"
    },
    valueSpinButtonDiv:{
        marginTop:"10px",
        display:"flex",
        flexDirection:"row"
    },
    minSpinBox:{
        width:"64px",
        height:"36px",
        paddingRight:"26px",
        marginLeft:"30px"
    },
    maxSpinBox:{
        width:"64px",
        height:"36px"
    },
    featureText:{
        width:"180px",
        height:"20px",
        marginTop:"1px",
        marginLeft:"30px",
        color:"#979797",
        textAlign:"left",
        position:"absolute"
    },
    treatCategorical:{
        width:"180px",
        height:"56px",
        marginTop:"21px",
        marginLeft:"30px",
        marginRight:"45px",
        marginBottom:"1px"
    },
    defaultText:{
        marginTop:"105px",
        marginRight:"35px",
        marginLeft:"35px",
        textAlign:"center",
        color: "#979797"
        
    }
});

export interface IFilterEditorState {
    openedFilter?: IFilter;
}

export class FilterEditor extends React.PureComponent<IFilterEditorProps, IFilterEditorState> {
    private _leftSelection: Selection;
    private readonly dataArray: IComboBoxOption[] = new Array(this.props.jointDataset.datasetFeatureCount).fill(0)
        .map((unused, index) => {
            const key = JointDataset.DataLabelRoot + index.toString();
            return {key, text: this.props.jointDataset.metaDict[key].abbridgedLabel}
        });

    private readonly leftItems = [
        JointDataset.IndexLabel,
        JointDataset.DataLabelRoot,
        JointDataset.PredictedYLabel,
        JointDataset.TrueYLabel,
        JointDataset.ClassificationError,
        JointDataset.RegressionError
    ].map(key => {
        const metaVal = this.props.jointDataset.metaDict[key];
        if (key === JointDataset.DataLabelRoot) {
            return {key, title: "Dataset"};
        }
        if  (metaVal === undefined) {
            return undefined;
        }
        return {key, title: metaVal.abbridgedLabel};
    }).filter(obj => obj !== undefined);

    private comparisonOptions: IComboBoxOption[] = [
        {
            key: FilterMethods.equal,
            text: localization.Filters.equalComparison
        },
        {
            key: FilterMethods.greaterThan,
            text: localization.Filters.greaterThanComparison
        },
        {
            key: FilterMethods.lessThan,
            text: localization.Filters.lessThanComparison
        },
        {
            key: FilterMethods.inTheRangeOf,
            text: localization.Filters.inTheRangeOf
        }
    ];
    private _isInitialized = false;

    constructor(props: IFilterEditorProps) {
        super(props);
        this.state = {openedFilter: this.props.initialFilter};
        this._leftSelection = new Selection({
            selectionMode: SelectionMode.single,
            onSelectionChanged: this._setSelection
          });
        this._leftSelection.setItems(this.leftItems);
        this._isInitialized = true;
    }

    componentDidUpdate(props:IFilterEditorProps, prevVal: IFilterEditorState) {
        if(this.props.initialFilter != undefined && this.props.initialFilter!=props.initialFilter){
            const newFilter = {
                arg: this.props.initialFilter.arg,
                column: this.props.initialFilter.column,
                method: this.props.initialFilter.method
            }
            this.setState({openedFilter: newFilter});
        }
    }
    
    public render(): React.ReactNode {
        const openedFilter = this.state.openedFilter;
        console.log("Here render", this.state.openedFilter);
        if(openedFilter == undefined)
        {  console.log("undefined state");
                return(
                    <div className={styles.wrapper}>
                            <div className={styles.leftHalf}>
                                <DetailsList
                                    className={styles.detailedList}
                                    items={this.leftItems}
                                    ariaLabelForSelectionColumn="Toggle selection"
                                    ariaLabelForSelectAllCheckbox="Toggle selection for all items"
                                    checkButtonAriaLabel="Row checkbox"
                                    checkboxVisibility={CheckboxVisibility.hidden}
                                    onRenderDetailsHeader={this._onRenderDetailsHeader}
                                    selection={this._leftSelection}
                                    selectionPreservedOnEmptyClick={false}
                                    setKey={"set"}
                                    columns={[{key: 'col1', name: 'name', minWidth: 150, fieldName: 'title'}]}
                                />
                            </div>
                            <div className={styles.rightHalf}> 
                            <Text className={styles.defaultText} variant={"medium"}>Select a filter to add parameters to your dataset cohort.</Text></div>
                    </div>
                );
        }
        else {
            console.log("defined state");
            const selectedMeta = this.props.jointDataset.metaDict[openedFilter.column];
            const numericDelta = selectedMeta.treatAsCategorical || selectedMeta.featureRange.rangeType === RangeTypes.integer ?
                1 : (selectedMeta.featureRange.max - selectedMeta.featureRange.min)/10;
            const isDataColumn = openedFilter.column.indexOf(JointDataset.DataLabelRoot) !== -1;
            let categoricalOptions: IComboBoxOption[] = [];
            if (selectedMeta.treatAsCategorical) {
                // Numerical values treated as categorical are stored with the values in the column,
                // true categorical values store indexes to the string values
                categoricalOptions = selectedMeta.isCategorical ? 
                    selectedMeta.sortedCategoricalValues.map((label, index) => {return {key: index, text: label}}) :
                    selectedMeta.sortedCategoricalValues.map((label) => {return {key: label, text: label.toString()}})
            }  
            return (
                    <div className={styles.wrapper}>
                        <div className={styles.leftHalf}>
                            <DetailsList
                                className={styles.detailedList}
                                items={this.leftItems}
                                ariaLabelForSelectionColumn="Toggle selection"
                                ariaLabelForSelectAllCheckbox="Toggle selection for all items"
                                checkButtonAriaLabel="Row checkbox"
                                checkboxVisibility={CheckboxVisibility.hidden}
                                onRenderDetailsHeader={this._onRenderDetailsHeader}
                                selection={this._leftSelection}
                                selectionPreservedOnEmptyClick={false}
                                setKey={"set"}
                                columns={[{key: 'col1', name: 'name', minWidth: 150, fieldName: 'title'}]}
                            />
                        </div>
                        <div className={styles.rightHalf}>
                                {isDataColumn && (
                                    <ComboBox
                                        className ={styles.featureComboBox}
                                        options={this.dataArray}
                                        onChange={this.setSelectedProperty}
                                        label={"Select Feature"}
                                        ariaLabel="feature picker"
                                        selectedKey={openedFilter.column}/>
                                )}
                                {selectedMeta.featureRange && selectedMeta.featureRange.rangeType === RangeTypes.integer && (
                                    <Checkbox className={styles.treatCategorical} label="Treat as categorical" checked={selectedMeta.treatAsCategorical} onChange={this.setAsCategorical} />
                                )}
                                {selectedMeta.treatAsCategorical && (
                                <div className={styles.featureTextDiv}>
                                    <Text variant={"small"}className={styles.featureText}>
                                        {`# of unique values: ${selectedMeta.sortedCategoricalValues.length}`}
                                    </Text>
                                    <ComboBox
                                        multiSelect
                                        label={localization.Filters.categoricalIncludeValues}
                                        className={styles.operationComboBox}
                                        selectedKey={openedFilter.arg}
                                        onChange={this.setCategoricalValues}
                                        options={categoricalOptions}
                                        useComboBoxAsMenuWidth={true}
                                    />
                                </div>
                                )}
                            {!selectedMeta.treatAsCategorical && (
                                <div className={styles.featureTextDiv}>
                                        <Text variant={"small"} className={styles.featureText}>
                                        {`Min: ${selectedMeta.featureRange.min}`} {`Average: ${selectedMeta.featureRange.min}`} {`Max: ${selectedMeta.featureRange.max}`}
                                        </Text>    
                                    <ComboBox
                                        label={localization.Filters.numericalComparison}
                                        className={styles.operationComboBox}
                                        selectedKey={openedFilter.method}
                                        onChange={this.setComparison}
                                        options={this.comparisonOptions}
                                        useComboBoxAsMenuWidth={true}
                                    />
                                    {openedFilter.method == FilterMethods.inTheRangeOf ? 
                                            <div className={styles.valueSpinButtonDiv}>
                                                <SpinButton
                                                    labelPosition={Position.top}
                                                    className ={styles.minSpinBox}
                                                    label={localization.Filters.minimum}
                                                    min={selectedMeta.featureRange.min}
                                                    max={selectedMeta.featureRange.max}
                                                />
                                                <SpinButton
                                                    labelPosition={Position.top}
                                                    className = {styles.maxSpinBox}
                                                    label={localization.Filters.maximum}
                                                    min={selectedMeta.featureRange.min}
                                                    max={selectedMeta.featureRange.max}
                                                />
                                            </div>
                                        :
                                            <div className={styles.valueSpinButtonDiv}>
                                                <SpinButton
                                                    labelPosition={Position.top}
                                                    className={styles.valueSpinButton}
                                                    label={localization.Filters.numericValue}
                                                    min={selectedMeta.featureRange.min}
                                                    max={selectedMeta.featureRange.max}
                                                    value={openedFilter.arg.toString()}
                                                    onIncrement={this.setNumericValue.bind(this, numericDelta, selectedMeta)}
                                                    onDecrement={this.setNumericValue.bind(this, -numericDelta, selectedMeta)}
                                                    onValidate={this.setNumericValue.bind(this, 0, selectedMeta)}
                                                />
                                            </div>
                                    }
                                </div>
                            )}
                            <DefaultButton 
                                className = {styles.addFilterButton}
                                text={"Add Filter"}
                                onClick={this.saveState}
                            />
                        </div>
                    </div>
            );
        }
    }

    private extractSelectionKey(key: string): string {
        let index = key.indexOf(JointDataset.DataLabelRoot);
        if (index !== -1) {
            return JointDataset.DataLabelRoot;
        }
        return key;
    }

    private readonly setAsCategorical = (ev: React.FormEvent<HTMLElement>, checked: boolean): void => {
        const openedFilter = this.state.openedFilter;
        this.props.jointDataset.setTreatAsCategorical(openedFilter.column, checked);
        if (checked) {
          this.setState(
              {
                  openedFilter:{
                      arg:[],
                      method:FilterMethods.includes,
                      column:openedFilter.column
                  }
              }
          )
        } else {

            this.setState(
                {
                    openedFilter:{
                        arg:this.props.jointDataset.metaDict[openedFilter.column].featureRange.max,
                        method:FilterMethods.includes,
                        column:FilterMethods.lessThan 
                    }
                }
            )
        }
    }

    private readonly _setSelection = (): void => {
        if (!this._isInitialized) {
            return;
        }
        let property = this._leftSelection.getSelection()[0].key as string;
        if (property === JointDataset.DataLabelRoot) {
            property += "0";
        }
        this.setDefaultStateForKey(property);
    }

    private readonly setSelectedProperty = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        const property = item.key as string;
        this.setDefaultStateForKey(property);
    }

    private saveState = (): void => {
        this.props.onAccept(this.state.openedFilter);
        this.setState({openedFilter:undefined});
    }

    private readonly _onRenderDetailsHeader = () => {
        return <div className={styles.filterHeader}>Select your filters</div>
    }

    private readonly setCategoricalValues = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        const openedFilter = this.state.openedFilter;
        const selectedVals = [...(openedFilter.arg as number[])];

        const index = selectedVals.indexOf(item.key as number);
        if (item.selected && index === -1) {
            selectedVals.push(item.key as number);
        } else {
            selectedVals.splice(index, 1);
        }
        this.setState(
            {
                openedFilter:{
                    arg:selectedVals,
                    method:openedFilter.method,
                    column:openedFilter.column 
                }
            }
        )
    }

    private readonly setComparison = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
       const openedFilter = this.state.openedFilter;
        this.setState(
            {
                openedFilter:{
                    arg:openedFilter.arg,
                    method:item.key as FilterMethods,
                    column:openedFilter.column 
                }
            }
        )
    }

    private readonly setNumericValue = (delta: number, column: IJointMeta, stringVal: string): string | void => {
        const openedFilter = this.state.openedFilter;
        if (delta === 0) {
            const number = +stringVal;
            if ((!Number.isInteger(number) && column.featureRange.rangeType === RangeTypes.integer)
                || number > column.featureRange.max || number < column.featureRange.min) {
                return this.state.openedFilter.arg.toString();
            }

            this.setState(
                {
                    openedFilter:{
                        arg:number,
                        method:openedFilter.method,
                        column:openedFilter.column 
                    }
                }
            )
        } else {
            const prevVal = openedFilter.arg as number;
            const newVal = prevVal + delta;
            if (newVal > column.featureRange.max || newVal < column.featureRange.min) {
                return prevVal.toString();
            }
            this.setState(
                {
                    openedFilter:{
                        arg:newVal,
                        method:openedFilter.method,
                        column:openedFilter.column 
                    }
                }
            )
        }
    }

    private setDefaultStateForKey(key: string): void {
        const openedFilter = this.state.openedFilter;
        let filter: IFilter = {column : key} as IFilter;
        const meta = this.props.jointDataset.metaDict[key];
        if (meta.isCategorical) {
            filter.method = FilterMethods.includes;
            filter.arg = Array.from(Array(meta.sortedCategoricalValues.length).keys());
        } else if(meta.treatAsCategorical) {
            filter.method = FilterMethods.includes;
            filter.arg = meta.sortedCategoricalValues as any[];
        } else {
            filter.method = FilterMethods.lessThan;
            filter.arg = meta.featureRange.max;
        }
        this.setState(
            {
                openedFilter:filter
            }
        )
    }
}