import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../../service/ImplementationService/projet.service'
import { HttpClient } from '@angular/common/http'
import { User } from 'src/app/service/Interfaces/user';
import { Router } from '@angular/router';
import { AuthService } from 'src/app/service/auth/auth.service';

@Component({
  selector: 'app-profil',
  templateUrl: './profil.component.html',
  styleUrls: ['./profil.component.css'],
  providers: [AuthService]
})
export class ProfilComponent implements OnInit {
  user:User
  id:number = JSON.parse(localStorage.getItem('user')).userId
  constructor(private projetService: ProjetService, private http: HttpClient,private router: Router,private authService: AuthService) { }

  ngOnInit(): void {
    this.getUser(this.id);
    
    
  }
  getUser(id) : void{
    this.projetService.getUser(id).subscribe(user => (this.user = user))
  }

  getLogin() {
    return this.authService.getUser().login;
  }

  logout() {
    return this.authService.logout();
  }
}
