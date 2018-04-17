package pricer

import (
	"net"
	"time"
	"fmt"
)

type IPC struct {
	conn *net.UnixConn
}

func NewIPC() (*IPC, error) {

	ipc := &IPC{}
	network := "unix"
	localName := fmt.Sprintf("mercury-%d.ipc", time.Now().UnixNano())
	lAddr := &net.UnixAddr{Name: localName, Net: network}
	rAddr := &net.UnixAddr{Name: "pricer.ipc", Net: network}
	conn, err := net.DialUnix(network, lAddr, rAddr)
	if err == nil {
		ipc.conn = conn
	} else {
		ipc = nil
	}
	return ipc, err
}

func (ipc *IPC) Write(bytes []byte) error {
	_, err := ipc.conn.Write(bytes)
	return err
}

func (ipc *IPC) Read(b [] byte) error {
	_, err := ipc.conn.Read(b)
	return err
}

func (ipc *IPC) Close() {
	ipc.conn.Close()
}
