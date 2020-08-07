import { Component, OnInit } from '@angular/core';
import { PriceService } from 'src/app/service/PriceService/price.service';
import { AuthService } from 'src/app/service/auth/auth.service';
@Component({
  selector: 'ngbd-dropdown-navbar',
  templateUrl: './dropdown-navbar.component.html',
  styleUrls: ['./dropdown-navbar.component.css']
})
export class DropdownNavbarComponent implements OnInit {
  collapsed = true;
  message:string;

  constructor(private price: PriceService ,private authService: AuthService) { }

  ngOnInit() {
    this.price.currentMessage.subscribe(message => this.message = message) ; 

  }

  newMessageDollar() {
    this.price.changeMessage("Dollar")
  }
  newMessageDH() {
    this.price.changeMessage("DH")
  }
  newMessageEuro() {
    this.price.changeMessage("Euro")
  }
  logout() {
    return this.authService.logout();
  }
}
