import React from 'react';
import MercuryTable from "../../components/table";
import { Input, Icon, Row, Col, Button } from 'antd';
import { connect, select } from 'dva';

class AmericanPricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>
                <MercuryTable
                    header="American Pricer"
                    namespace="american"
                    addStock={false}
                    columns={this.props.american.columns}
                    dataSource={this.props.american.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )
    }
}

export default connect(({ american }) => ({ american }))(AmericanPricer);
