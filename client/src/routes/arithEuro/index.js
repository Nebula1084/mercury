import React from 'react';
import MercuryTable from '../../components/table';
import { Input, Icon, Row, Col, Button } from 'antd';
import styles from './european.less';
import { connect, select } from 'dva';

class ArithEuroPricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>
                <MercuryTable
                    header="Arithmetic Basket European Pricer"
                    namespace="arithEuro"
                    closedForm={false}
                    useGpu={true}
                    controlVariate={true}
                    addStock={true}
                    columns={this.props.arithEuro.columns}
                    dataSource={this.props.arithEuro.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )
    }
}

export default connect(({ arithEuro }) => ({ arithEuro }))(ArithEuroPricer);
