import React from 'react'
import './Card.css'

export default function Card(props) {
    let cn1, cn2;
    if(props.name === "FLASK" || props.name === "Light Intensity Sensor"){
        cn1 = "card bg-slate-300"
        cn2 = "header text-blue-600"
    }
    else{
        cn1 = "card"
        cn2 = "header"
    }
      
  return (
      <div className={cn1}>
  <img src= {props.img} alt="" className='h-24'/>
  <div className={cn2}>{props.name}</div>
</div>
  )
}
