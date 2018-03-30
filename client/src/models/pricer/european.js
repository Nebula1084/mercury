import { load, update, edit, save, cancel } from "./util.js"
export default {
    namespace: 'european',
    state: {
        rows: [],
        stash: []
    },
    reducers: {
        import: load,
        add(state) {

        },
        update: update,
        edit: edit,
        save: save,
        cancel: cancel
    }
}