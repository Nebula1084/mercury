import { Button, Icon } from 'antd'

export default class ImportButton extends React.Component {
    render() {
        return (
            <div >
                <input ref="fileInput" type="file" onChange={this.props.onChange} style={{ display: "none" }} />
                <Button onClick={() => {
                    this.refs.fileInput.value = "";
                    this.refs.fileInput.click()
                }}>
                    <Icon type="upload" /> Import
                </Button>
            </div >
        )
    }
}