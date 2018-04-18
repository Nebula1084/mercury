import React from 'react';
import MercuryTable from '../../components/table';
import { Input, Icon, Row, Col, Button } from 'antd';
import styles from './european.less';
import { connect, select } from 'dva';

class GeoEuroPricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>
                <MercuryTable
                    header="Geometric Basket European Pricer"
                    namespace="geoEuro"
                    closedForm={true}
                    useGpu={true}
                    addStock={true}
                    columns={this.props.geoEuro.columns}
                    dataSource={this.props.geoEuro.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )
    }
}

export default connect(({ geoEuro }) => ({ geoEuro }))(GeoEuroPricer);
