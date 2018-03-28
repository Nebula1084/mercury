import React from 'react';
import { connect } from 'dva';
import { Table, Icon, Switch, Radio, Form, Row, Col } from 'antd';
import styles from './table.less';

const expandedRowRender = record => <p>{record.description}</p>;
const title = () => 'Here is title';
const showHeader = true;
const footer = () => 'Here is footer';
const scroll = { y: 250 };

export default class MercuryTable extends React.Component {

  state = {
    bordered: false,
    loading: false,
    pagination: true,
    size: 'default',    
    title,
    showHeader,
    footer,    
    scroll: undefined,
  }

  constructor(props){
    super(props)
    
  }

  componentDidMount() {
    // const { dispatch } = this.props;
    // dispatch({ type: 'table/query' });
  }

  render() {
    const state = this.state;

    return (
        <div className={styles['showcase-container']}>
          <Table {...this.state} columns={this.props.columns} dataSource={this.props.dataSource} />
        </div>      
    );
  }
}