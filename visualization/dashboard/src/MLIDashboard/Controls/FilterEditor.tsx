import React from "react";
import { IJointMeta, JointDataset } from "../JointDataset";
import { Button, PrimaryButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { SpinButton } from 'office-ui-fabric-react/lib/SpinButton';
import { FilterMethods, IFilter } from "../Interfaces/IFilter";
import { IComboBoxOption, IComboBox, ComboBox } from "office-ui-fabric-react/lib/ComboBox";
import { FabricStyles } from "../FabricStyles";
import { localization } from "../../Localization/localization";
import { RangeTypes } from "mlchartlib";
import { Target} from "office-ui-fabric-react/lib/Callout";
import _ from "lodash";
import { mergeStyleSets, FontSizes, fontFace, ThemeSettingName} from "@uifabric/styling";
import { DetailsList, Selection, SelectionMode, IColumn } from "office-ui-fabric-react/lib/DetailsList";
import { Checkbox } from "office-ui-fabric-react/lib/Checkbox";
import { defaultTheme } from "../Themes";
import { FontWeights, getTheme, ColorPicker, mergeAriaAttributeValues, resetControlledWarnings } from "office-ui-fabric-react";

export interface IFilterEditorProps {
    jointDataset: JointDataset;
    initialFilter: IFilter;
    target?: Target;
    onAccept: (filter: IFilter) => void;
    onCancel: () => void;
}

const theme = getTheme();

const styles = mergeStyleSets({
    wrapper: {
        minHeight: "300px",
        display: "flex",
        marginTop: "60px",
    },
    leftHalf: {
        display: "inline-flex",
        maxWidth: "50%",
        //width: "50%",
        height: "100%",
        //borderRight: "2px solid #CCC",
        margin:"auto"
    },
    rightHalf: {
        //margin: "auto",
        display: "inline-flex",
        minWidth: "50%",
        flexDirection: "column",
        background: "#F4F4F4",
        //borderRadius: "5px"
    },
    detailedList: {
        margin:"auto"
    },
    filterHeader:{
            fontWeight: FontWeights.semibold,
            fontSize: FontSizes.medium,
            color: "#000000"
        },
    dataSummary: {
        fontWeight: FontWeights.semibold,
        fontSize: FontSizes.medium,
        color: "#979797"
    }, 
    addFilterButton: {
        
    },
    minRangeOf:{
        display:"flex",
        flexDirection: "row",
        width:"64px",
        height:"36px",
        alignSelf:"flex-start"
    },
    maxRangeOf:{
        display:"flex",
        flexDirection: "row",
        width:"64px",
        height:"36px",
        alignSelf:"flex-end"
    },
    subDiv:{
        marginBottom:"1px",
        alignSelf:"center"
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
    //private _initialState:IFilter;

    constructor(props: IFilterEditorProps) {
        super(props);
        console.log("constructor component")
        //this.state = _.cloneDeep(this.props.initialFilter);
        
        this.state = {openedFilter: this.props.initialFilter};
        //this._initialState = this.state;
        console.log("constructor state", this.state)
        //console.log("constructor initial state", this._initialState)
        this._leftSelection = new Selection({
            selectionMode: SelectionMode.single,
            onSelectionChanged: this._setSelection
          });
        this._leftSelection.setItems(this.leftItems);
        if (this.props.initialFilter!=undefined){
            this._leftSelection.setKeySelected(this.extractSelectionKey(this.props.initialFilter.column), true, false); 
        }
        //this._leftSelection.setKeySelected(this.extractSelectionKey(this.props.initialFilter.column), true, false);
        this._isInitialized = true;
    }

    componentDidUpdate(props:IFilterEditorProps, prevVal: IFilterEditorState) {
       console.log("update component");
        if(this.props.initialFilter != undefined && this.props.initialFilter!=props.initialFilter){
            const newFilter = {
                arg: this.props.initialFilter.arg,
                column: this.props.initialFilter.column,
                method: this.props.initialFilter.method
            }
            this.setState({openedFilter: newFilter});  
            console.log("update component12", this.state.openedFilter);   
            
            console.log("update component new ", newFilter);     
        }
        else{
            console.log("this.state component", this.state);
            console.log("props component", props.initialFilter);
            console.log("prevVale component", prevVal);
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
                                    onRenderDetailsHeader={this._onRenderDetailsHeader}
                                    selection={this._leftSelection}
                                    selectionPreservedOnEmptyClick={false}
                                    setKey={"set"}
                                    columns={[{key: 'col1', name: 'name', minWidth: 150, fieldName: 'title'}]}
                                />
                            </div>
                            <div className={styles.rightHalf}> Select filter</div>
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
                                onRenderDetailsHeader={this._onRenderDetailsHeader}
                                selection={this._leftSelection}
                                selectionPreservedOnEmptyClick={false}
                                setKey={"set"}
                                columns={[{key: 'col1', name: 'name', minWidth: 150, fieldName: 'title'}]}
                            />
                        </div>
                        <div className={styles.rightHalf}>
                            <div>
                        {isDataColumn && (
                            <ComboBox
                                options={this.dataArray}
                                onChange={this.setSelectedProperty}
                                label={"Feature: "}
                                ariaLabel="feature picker"
                                selectedKey={openedFilter.column}
                                useComboBoxAsMenuWidth={true}
                                styles={FabricStyles.defaultDropdownStyle} />
                        )}
                        {selectedMeta.featureRange && selectedMeta.featureRange.rangeType === RangeTypes.integer && (
                            <Checkbox label="Treat as categorical" checked={selectedMeta.treatAsCategorical} onChange={this.setAsCategorical} />
                        )}
                        {selectedMeta.treatAsCategorical && (
                            <div className={styles.subDiv}>
                                <div className={styles.dataSummary}>{`# of unique values: ${selectedMeta.sortedCategoricalValues.length}`}</div>
                                <ComboBox
                                    multiSelect
                                    label={localization.Filters.categoricalIncludeValues}
                                    className="path-selector"
                                    selectedKey={openedFilter.arg}
                                    onChange={this.setCategoricalValues}
                                    options={categoricalOptions}
                                    useComboBoxAsMenuWidth={true}
                                    styles={FabricStyles.smallDropdownStyle}
                                />
                            </div>
                        )}
                        {!selectedMeta.treatAsCategorical && (
                                <div>
                                    <div className={styles.subDiv}>{`min: ${selectedMeta.featureRange.min}`} {`avg: ${selectedMeta.featureRange.min}`} {`max: ${selectedMeta.featureRange.max}`}</div>
                                <ComboBox
                                    label={localization.Filters.numericalComparison}
                                    className="path-selector"
                                    selectedKey={openedFilter.method}
                                    onChange={this.setComparison}
                                    options={this.comparisonOptions}
                                    useComboBoxAsMenuWidth={true}
                                    styles={FabricStyles.smallDropdownStyle}
                                />
                                {openedFilter.method == FilterMethods.inTheRangeOf ? 
                                        <div>
                                            <SpinButton
                                                className ={styles.minRangeOf}
                                                label={localization.Filters.minimum}
                                                min={selectedMeta.featureRange.min}
                                                max={selectedMeta.featureRange.max}
                                            />
                                            <SpinButton
                                                className = {styles.maxRangeOf}
                                                label={localization.Filters.maximum}
                                                min={selectedMeta.featureRange.min}
                                                max={selectedMeta.featureRange.max}
                                            />
                                        </div>
                                    :
                                        <div>
                                            <SpinButton
                                                styles={{
                                                    spinButtonWrapper: {maxWidth: "98px"},
                                                    labelWrapper: { alignSelf: "center"},
                                                    root: {
                                                        display: "inline-flex",
                                                        float: "right",
                                                        selectors: {
                                                            "> div": {
                                                                maxWidth: "108px"
                                                            }
                                                        }
                                                    }
                                                }}
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
           // this.setState({arg:[], method: FilterMethods.includes});
           //this.setState({openedFilter.arg:[], method: FilterMethods.includes});
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
            // this.setState(prevState => ({
            //     openedFilter: {                   // object that we want to update
            //         ...prevState.openedFilter,    // keep all other key-value pairs
            //         arg: this.props.jointDataset.metaDict[this.state.openedFilter.column].featureRange.max,
            //         method:FilterMethods.lessThan 
            //     }
            // }));
            // this.setState({
            //     arg:this.props.jointDataset.metaDict[this.state.column].featureRange.max,
            //     method: FilterMethods.lessThan
            // });
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

    // private setInitialState = (): void => {
    //     console.log("this._initialState", this._initialState);
    //     this.state = this._initialState;
    // }
    private saveState = (): void => {
        console.log("@@@@@@@@@@@@@@@@@@@@@@@@@@@");
        this.props.onAccept(this.state.openedFilter);
        console.log("%%%%%%%%%%%%%%%%%%%%%%%%%%");

        console.log("Save state", this.state);
        //this.props.onAccept(this.state);
        //const state = this.state;// = undefined;
        //this.setInitialState;
        //this.setState;
        this.setState({openedFilter:undefined});

        console.log("set state in save state", this.state);
        //this.forceUpdate();
    }

    private readonly _onRenderDetailsHeader = () => {
        return <div className={styles.filterHeader}>Select your filters</div>
    }

    private readonly setCategoricalValues = (event: React.FormEvent<IComboBox>, item: IComboBoxOption): void => {
        //const selectedVals = [...(this.state.arg as number[])];
        //const selectedVals = [...(this.state.openedFilter.arg as number[])];
        const openedFilter = this.state.openedFilter;
        const selectedVals = [...(openedFilter.arg as number[])];

        const index = selectedVals.indexOf(item.key as number);
        if (item.selected && index === -1) {
            selectedVals.push(item.key as number);
        } else {
            selectedVals.splice(index, 1);
        }

        // this.setState(prevState => ({
        //     openedFilter: {                   // object that we want to update
        //         ...prevState.openedFilter,    // keep all other key-value pairs
        //         arg: selectedVals      // update the value of specific key
        //     }
        // }))

        this.setState(
            {
                openedFilter:{
                    arg:selectedVals,
                    method:openedFilter.method,
                    column:openedFilter.column 
                }
            }
        )

       // this.setState({arg: selectedVals});
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

        // this.setState(prevState => ({
        //     openedFilter: {                   // object that we want to update
        //         ...prevState.openedFilter,    // keep all other key-value pairs
        //         method: item.key as FilterMethods      // update the value of specific key
        //     }
        // }))
        //this.setState({method: item.key as FilterMethods});
    }

    private readonly setNumericValue = (delta: number, column: IJointMeta, stringVal: string): string | void => {
        const openedFilter = this.state.openedFilter;
        if (delta === 0) {
            const number = +stringVal;
            if ((!Number.isInteger(number) && column.featureRange.rangeType === RangeTypes.integer)
                || number > column.featureRange.max || number < column.featureRange.min) {
                return this.state.openedFilter.arg.toString();
                //return this.state.arg.toString();
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
    
            // this.setState(prevState => ({
            //     openedFilter: {                   // object that we want to update
            //         ...prevState.openedFilter,    // keep all other key-value pairs
            //         arg: number      // update the value of specific key
            //     }
            // }))
            //this.setState({arg: number});
        } else {
            //const prevVal = this.state.arg as number;
            //const prevVal = this.state.openedFilter.arg as number;
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


            // this.setState(prevState => ({
            //     openedFilter: {                   // object that we want to update
            //         ...prevState.openedFilter,    // keep all other key-value pairs
            //         arg: newVal      // update the value of specific key
            //     }
            // }))
            //this.setState({arg: newVal});
        }
    }

    private setDefaultStateForKey(key: string): void {
        const openedFilter = this.state.openedFilter;
        console.log("filter key", key);
        let filter: IFilter = {column : key} as IFilter;
        //debugger;
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


        // this.setState(prevState => ({
        //     openedFilter: {                   // object that we want to update
        //         ...prevState.openedFilter,    // keep all other key-value pairs
        //         column: key      // update the value of specific key
        //     }
        // }))
        //this.setState(filter);
    }
}