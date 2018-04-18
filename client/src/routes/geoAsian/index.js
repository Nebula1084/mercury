import React from 'react';
import MercuryTable from '../../components/table';
import PricingMonitor from '../../components/monitor'
import { Input, Icon, Row, Col, Button } from 'antd';
import { connect, select } from 'dva';

class GeoAsianPricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {

        return (
            <div>
                <MercuryTable
                    header="Geometric Asian Pricer"
                    namespace="geoAsian"
                    closedForm={true}
                    useGpu={true}
                    addStock={false}
                    columns={this.props.geoAsian.columns}
                    dataSource={this.props.geoAsian.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )

    }
}

export default connect(({ geoAsian }) => ({ geoAsian }))(GeoAsianPricer);
