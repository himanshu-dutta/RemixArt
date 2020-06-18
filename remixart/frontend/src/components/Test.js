import React, { Component } from "react";
import { Button, Badge } from "reactstrap";

class Test extends Component {
  constructor(props) {
    super(props);

    this.state = {
      user: "Archita",
    };
  }
  changeUser() {
    this.setState({
      user: this.state.user === "Himanshu" ? "Archita" : "Himanshu",
    });
  }
  render() {
    return (
      <div>
        <Badge color="secondary">
          <h1>{this.state.user}'s App</h1>
        </Badge>
        <br />
        <Button color="info" onClick={() => this.changeUser()}>
          Alter
        </Button>
      </div>
    );
  }
}

export default Test;
