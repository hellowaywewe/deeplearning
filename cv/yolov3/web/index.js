// 一、获取预定义参数
var http =require("http");
var fs = require("fs");
var url = require("url");
var cv = require('opencv4nodejs')
var image = cv.imread('11.jpg')
var cvtGImage = image.cvtColor(cv.COLOR_BGR2GRAY)
cv.imwrite('去色.jpg', cvtGImage)
var thresholdImage = cvtGImage.adaptiveThreshold(255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
cv.imwrite('自适应阈值.jpg', thresholdImage)

function isImage(ext) {
    return [‘png’, ‘jpg’, ‘jpeg’].indexOf(ext.toLowerCase()) !== -1;
}

function readImage(img_path){

}

  const width = im.width();
  const height = im.height();

  if (width < 1 || height < 1) {
    throw new Error('Image has no size');
  }

  // do some cool stuff with img

  // save img
  img.save('./img/myNewImage.jpg');
});



var formidable = require("formidable");
    // 二、创建服务器，端口号
var server = http.createServer(function(req,res){
var pathname = url.parse(req.url,true).pathname;
    // 三、显示HTML样式
if(pathname == "/"){
    var rs = fs.createReadStream("index.html");
    rs.pipe(res);
}
// 四、点击提交，请求传递
else if(pathname == "/predict"){
        // 1、引入文件
        var form = new formidable.IncomingForm();
        // 2、解析：获取 字段、文件
        form.parse(req,function(err,fields,files){
            if(err){
                return console.log(err);
            }else{
                // 2.1 存储字段
                var fieldStr = JSON.stringify(fields);
                fs.writeFileSync("1.txt",fieldStr);
                // 2.2 存储文件(图片) 
                // 2.2.1、判断是否有装图片的文件夹，无则创建一个
                if(!fs.existsSync("uploads")){
                    fs.mkdir("uploads");
                }
                // 2.2.2、获取图片临时存放路径
                var filePath = files.img.path;
                // 2.2.3、创建可读、可写流;将图片写入文件夹
                var rs = fs.createReadStream(filePath);
                var ws = fs.createWriteStream("/root/workspaces/dataset/shanshui/images" + files.img.path);
                rs.pipe(ws);
                // 2.2.4、判断是否存放成功 (可读流结束)
                res.setHeader("Content-type","text/html;charset=utf-8");
                rs.on("end",function(){
                    res.write("上传成功");
                    res.end();
                })
            }
        })
    }
})
server.listen(8081);
