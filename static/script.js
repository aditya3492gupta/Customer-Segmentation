function handleSubmit(event) {
    event.preventDefault(); // prevent the form from submitting normally
    var file = document.getElementById("file-upload").files[0];
    var formData = new FormData();
    formData.append("file", file);
    var spinner = document.getElementById("spinner");
    spinner.style.display = "block";
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/predict");
    xhr.send(formData);
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            document.getElementById("results").innerHTML = "";
            var response = JSON.parse(xhr.responseText);

            var amountImg = document.createElement("img");
            amountImg.src = response.amount_img;
            amountImg.style.width = "30%";
            var freqImg = document.createElement("img");
            freqImg.src = response.freq_img;
            freqImg.style.width = "30%";
            var recencyImg = document.createElement("img");
            recencyImg.src = response.recency_img;
            recencyImg.style.width = "30%";

            spinner.style.display = "none";
            var imagesDiv = document.createElement("div");
            imagesDiv.style.display = "flex";
            imagesDiv.style.flexWrap = "wrap";
            imagesDiv.style.marginTop = "20px";
            imagesDiv.style.marginBottom = "20px";
            imagesDiv.style.justifyContent = "space-between";
            imagesDiv.style.alignItems = "center";
            imagesDiv.style.width = "100%";
            imagesDiv.appendChild(amountImg);
            imagesDiv.appendChild(freqImg);
            imagesDiv.appendChild(recencyImg);
            document.getElementById("results").appendChild(imagesDiv);
        }
    };
}