import React from 'react';

export default (props) => {

  const style = {
    width: props.size + '%',
    height: props.size + '%',
    left: props.dot[0] + '%',
    top: props.dot[1] + '%'
  }

  return (
    <div className="snake-food" style={style}></div>
  )
}