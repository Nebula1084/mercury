package main

import (
	"github.com/labstack/echo"
	"golang.org/x/net/websocket"
	"time"
	"fmt"
	"net/http"
	"github.com/Nebula1084/mercury/server/pricer"
)

// API is a defined as struct bundle
// for api. Feel free to organize
// your app as you wish.
type API struct{}

func IndexHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "hello world")
}

// Bind attaches api routes
func (api *API) Bind(group *echo.Group) {
	group.GET("/stream", api.StreamHandler)
	group.POST("/price", api.PriceHandler)
}

func (api *API) PriceHandler(c echo.Context) error {

	protocol := new(pricer.Protocol)
	if err := c.Bind(protocol); err != nil {
		fmt.Println(err.Error())
	}

	res, err := protocol.Call()
	if err != nil {
		fmt.Println(err.Error())
		res = &pricer.Result{Mean: -1, Conf: -1}
	}
	return c.JSON(200, res)
}

func (api *API) StreamHandler(c echo.Context) error {

	websocket.Handler(func(ws *websocket.Conn) {
		fmt.Printf("Socket incoming\n")
		elapsed := 1
		go func() {
			msg := ""
			websocket.Message.Receive(ws, msg)
			fmt.Printf("Close:%s\n", msg)
			ws.Close()
		}()
		for {
			fmt.Printf("Send\n")
			value := 2.
			u := value + (1.0 / float64(elapsed))
			l := value - (1.0 / float64(elapsed))
			msg := fmt.Sprintf("{\"value\":%f, \"u\":%f, \"l\":%f, \"time\":%d}", value, u, l, elapsed)
			println(msg)
			err := websocket.Message.Send(ws, msg)
			if err != nil {
				c.Logger().Error(err)
				break
			}
			time.Sleep(1 * time.Second)
			elapsed += 1
		}
		fmt.Printf("Socket end\n")
	}).ServeHTTP(c.Response(), c.Request())
	return nil
}
