function theimage(){
    var filename = document.getElementById('file-id').value;
    document.getElementById('filePath').innerHTML = filename;
}
var myConfig = {
    type: 'wordcloud',
    options: {
        text: 'jhbvbkfnah',
    }
};

zingchart.render({
    id: 'myChart',
    data: myConfig,
    height: 400,
    width: '100%'
});

function writeFile() {
    var fs = require('fs');

    fs.appendFile('mynewfile1.txt', ' This is my text.', function (err) {
        if (err) throw err;
        console.log('Updated!');
    });

}
