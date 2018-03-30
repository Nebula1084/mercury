import { Button, Icon } from 'antd'
import styles from './table.less';

export default class ImportButton extends React.Component {
    render() {
        return (
            <div >
                <input ref="fileInput" type="file" onChange={this.props.onChange} style={{ display: "none" }} />
                <Button className={styles['tool-button']} onClick={() => {
                    this.refs.fileInput.value = "";
                    this.refs.fileInput.click()
                }}>
                    <Icon type="file-add" /> Import
                </Button>
            </div >
        )
    }
}