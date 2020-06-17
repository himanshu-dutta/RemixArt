import React, { Component } from "react";

class Header extends Component {
  constructor(props) {
    super(props);

    this.state = {
      userName: "Archita",
    };
  }
  changeUserName() {
    this.setState({
      userName: this.state.userName === "Himanshu" ? "Archita" : "Himanshu",
    });
  }

  render() {
    return (
      <div>
        <div className="bg-primary text-white text-center p-2">
          <h1>{this.state.userName}'s Web App</h1>
          <button
            className="btn btn-primary m-2"
            onClick={() => this.changeUserName()}
          >
            Change
          </button>
        </div>
      </div>
    );
  }
}

export default Header;
