var database = firebase.database();
const alarmContainer =document.getElementById("tbody");

let ref = database.ref("tasks");

ref.on("value", gotData, errData);

function gotData(data){
    alarmContainer.innerHTML=``;
    x = data.val();
    count=1;
    value=true;
    for (n in x){
        dato= x[n]
        if(value){
            alarmContainer.innerHTML+=`<tr class="table-active">
                <td scope="row">${count}</td>
                <td scope="row">${dato.fecha}</td>
                <td scope="row">${dato.hora}</td>
                <td scope="row"> <img src="data:image/jpeg;base64,${dato.imagen}" width="200" 
                height="200" alt=""></td>
            </tr>`
            value=false;
        }else{
            alarmContainer.innerHTML+=`<tr class="table-primary">
                <td scope="row">${count}</td>
                <td scope="row">${dato.fecha}</td>
                <td scope="row">${dato.hora}</td>
                <td scope="row"> <img src="data:image/jpeg;base64,${dato.imagen}" width="200" 
                height="200" alt=""></td>
            </tr>`
            value=true;
        }
    count+=1;
    }
}
function errData(err){
    console.log("Error");
    console.log(err);
}