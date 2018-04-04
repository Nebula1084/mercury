package main

import (
	"github.com/labstack/echo"
	"golang.org/x/net/websocket"
	"time"
	"fmt"
	"net/http"
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
	group.GET("/v1/conf", api.ConfHandler)
	group.GET("/geometric", api.GeometricHandler)
}

// ConfHandler handle the app config, for example
func (api *API) ConfHandler(c echo.Context) error {
	app := c.Get("app").(*App)
	return c.JSON(200, app.Conf.Root)
}

func (api *API) GeometricHandler(c echo.Context) error {

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
