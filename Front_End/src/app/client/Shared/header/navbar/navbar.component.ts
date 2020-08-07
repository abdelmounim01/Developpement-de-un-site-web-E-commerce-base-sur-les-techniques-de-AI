import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../../../../service/ImplementationService/projet.service'
import { Router } from '@angular/router'
import { User } from '../../../../service/Interfaces/user';
import { AuthService } from 'src/app/service/auth/auth.service';
import { PriceService } from 'src/app/service/PriceService/price.service';

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})
export class NavbarComponent implements OnInit {
  collapsed = true;
  exist:any
  users: User[]
  model: any = {};
  user:User
  id:number = JSON.parse(localStorage.getItem('user')).userId
  message:string;

  constructor(private price: PriceService , private _auth: ProjetService, private _router: Router,private authService: AuthService) { }



  ngOnInit() {
    this.getUser(this.id);
    
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

  
  getUser(id) : void{
    this._auth.getUser(id).subscribe(user => (this.user = user))
  }
  logout() {
    return this.authService.logout();
  }

getusers(){
  this._auth.getUsers().subscribe(users =>{
    this.users = users
    console.log(this.users);
    
  })

}



}