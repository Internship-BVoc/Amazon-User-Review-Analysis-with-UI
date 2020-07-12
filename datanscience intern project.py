from flask import Flask, render_template
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
def home():
    return """<!DOCTYPE html>
<head>
<title>\Product Analyser</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {
  box-sizing: border-box;
}

body {
  font-family: Arial, Helvetica, sans-serif;
  font-weight:bold;
  font-variant:small-caps;
  margin: 0;
}

/* Style the header */
.header {
  padding: 40px;
  text-align: center;
  background: yellow;
  color: black;
}

/* Increase the font size of the h1 element */
.header h1 {
  font-size: 40px;
}

/* Style the top navigation bar */
.navbar {
  overflow: hidden;
  background-color: black;
}

/* Style the navigation bar links */
.navbar a {
  float: left;
  display: block;
  color: white;
  text-align: center;
  padding: 14px 20px;
  text-decoration: none;
}

/* Right-aligned link */
.navbar a.right {
  float: right;
}

/* Change color on hover */
.navbar a:hover {
  background-color: red;
  color: black;
}

/* Column container */
.row {
  display: flex;
  flex-wrap: wrap;

}

/* Create two unequal columns that sits next to each other */
/* Sidebar/left column */
.side {
  flex: 30%;
  background-color: #f1f1f1;
  padding: 20px;
}

/* Main column */
.main {
  flex: 70%;
  background-color: white;
  padding: 20px;
}

/* Fake image, just for this example */
.fakeimg {
  background-color: #aaa;
  width: 100%;
  padding: 15px;
}

/* Responsive layout - when the screen is less than 700px wide, make the two columns stack on top of each other instead of next to each other */
@media screen and (max-width: 700px) {
  .row {
    flex-direction: column;
  }
}

/* Responsive layout - when the screen is less than 400px wide, make the navigation links stack on top of each other instead of next to each other */
@media screen and (max-width: 400px) {
  .navbar a {
    float: none;
    width:100%;
  }
}
textarea.myTextBox{
 resize:'none';
 border:"none";
 display:block;
 margin-left:auto;
 margin-right:auto;

 }
</style>
</head>
<body>

<div style="background:">
	<h4 style="text-align:left; font-variant: small-caps;">Parsing the product and user review data.</h4>
</div>

<div class="navbar">
  <a href="#">Home</a>
  <a href="#">About</a>
  <a href="#" class="right">Feedback</a>
</div>

<div class="header">
  <h1>PRODUCT  ANALYSER</h1>
  <p>Analyse the Product by user Review based on the Sentiment</p>
</div>

 
  
 
    <style> 
 
          
 
        /* styling navlist */ 
 
        #navlist { 
 
            background-color: green; 
 
            position: absolute; 
 
            width: 100%; 
 
        } 
 
          
 
        /* styling navlist anchor element */ 
 
        #navlist a { 
 
            float:left; 
 
            display: block; 
 
            color: black; 
 
            text-align: center; 
 
            padding: 12px; 
 
            text-decoration: none; 
 
            font-size: 15px; 
 
        } 
 
        .navlist-right{ 
 
            float:right; 
 
        } 
 
  
 
        /* hover effect of navlist anchor element */ 
 
        #navlist a:hover { 
 
            background-color: red; 
 
            color: black; 
 
        } 
 
      
 
    </style> 
 
</head> 
 
 
<body> 
 
      
 
    <!-- Navbar items -->
 
    <div id="navlist"> 
 
        <a href="#">Home</a> 
 
        <a href="#">Our Products</a>
 
          
 
                <button> 
 
                    
                </button> 
 
            </form> 
 
        </div> 
 
    </div> 
	
	
      
 
    <!-- logo with tag -->
 
    
	<html lang="en">
<head>

<html lang="en">
<head>

<textarea name="myTextBox" cols="100 rows="5" style="background-color:#FCF5D8;color:#AD8C08;border:3px solid #AD8C08;">
Enter some text...
</textarea>
<br />
<input type="submit" />
</form>

<html lang="en">
<head>
  <title>Bootstrap Example</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>
<body>


<div class="container">
<div class="btn-group-vertical">
<button type="button" class="btn btn-primary">Text Analyze</button>
<button type="button" class="btn btn-primary">Emotions</button>
<button type="button" class="btn btn-primary">Entities Prediction</button>
<button type="button" class="btn btn-primary">Category Detection</button>
</div>
</div>

</body>
</html>


</body>
</html>


</body>
</html>

</body>
</html>

</body>
</html>
 
</body> 
 
  
 
 
</html>"""


if __name__ == '__main__':
    app.run(debug=True)

