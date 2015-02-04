from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import app

serverport = 80
http_server = HTTPServer(WSGIContainer(app.app))

http_server.listen(serverport)
IOLoop.instance().start() 
